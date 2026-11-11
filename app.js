const els = {
  canvas: document.getElementById("canvas"),
  video: document.getElementById("video"),
  placeholder: document.getElementById("placeholder"),
  placeholderText: document.getElementById("placeholderText"),
  stage: document.getElementById("stage"),
  hud: document.getElementById("hud"),
  fps: document.getElementById("fps"),
  hudMode: document.getElementById("hudMode"),
  conf: document.getElementById("conf"),
  confVal: document.getElementById("confVal"),
  pickBtn: document.getElementById("pickBtn"),
  camBtn: document.getElementById("camBtn"),
  fileInput: document.getElementById("fileInput"),
  statusCard: document.getElementById("statusCard"),
  statusText: document.getElementById("statusText"),
  statusHint: document.getElementById("statusHint"),
  lampRed: document.getElementById("lampRed"),
  lampYellow: document.getElementById("lampYellow"),
  lampGreen: document.getElementById("lampGreen"),
  latency: document.getElementById("latency"),
  count: document.getElementById("count"),
  detList: document.getElementById("detList"),
  summaryCounts: document.getElementById("summaryCounts"),
  modelBadge: document.getElementById("modelBadge"),
  toast: document.getElementById("toast"),
  tabs: Array.from(document.querySelectorAll(".tab")),
};

const COLORS = { red: "#ff3b30", yellow: "#ffcc00", green: "#34c759", off: "#8e8e93" };
const NAMES = { red: "红灯", yellow: "黄灯", green: "绿灯", off: "熄灭" };
const HINTS = { red: "禁止通行", yellow: "注意减速", green: "可以通行", off: "信号灯未点亮" };
const INPUT_SIZE = 640;

const state = {
  session: null,
  ready: false,
  mode: "image",
  conf: 0.35,
  imageCanvas: null,
  stream: null,
  running: false,
  busy: false,
  dets: [],
  frameWidth: 0,
  frameHeight: 0,
  frames: 0,
  fpsTimer: performance.now(),
  loopTimer: null,
};

const ctx = els.canvas.getContext("2d");
const inputCanvas = document.createElement("canvas");
inputCanvas.width = INPUT_SIZE;
inputCanvas.height = INPUT_SIZE;
const inputCtx = inputCanvas.getContext("2d", { willReadFrequently: true });
const inputFloat = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

function toast(message) {
  els.toast.textContent = message;
  els.toast.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => els.toast.classList.remove("show"), 3000);
}

function setBadge(text) {
  els.modelBadge.textContent = text;
}

function setConf(value) {
  state.conf = Number(value);
  els.confVal.textContent = state.conf.toFixed(2);
}

function showPlaceholder(show) {
  els.placeholder.classList.toggle("hidden", !show);
  els.canvas.classList.toggle("hidden", show);
}

/* ---------------------------- model loading ---------------------------- */

async function fetchModel(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("模型下载失败 HTTP " + res.status);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) {
    const buf = new Uint8Array(await res.arrayBuffer());
    onProgress(1);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received / total);
  }
  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

async function initModel() {
  if (typeof ort === "undefined") {
    throw new Error("推理库加载失败，请刷新页面");
  }
  setBadge("初始化推理引擎…");
  ort.env.wasm.wasmPaths = new URL("vendor/ort/", document.baseURI).href;
  ort.env.wasm.simd = true;
  const threads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ort.env.wasm.numThreads = threads;
  ort.env.logLevel = "error";

  const modelBytes = await fetchModel("models/traffic_light.onnx", (p) => {
    setBadge(`下载模型 ${Math.round(p * 100)}%`);
  });

  setBadge("创建推理会话…");
  state.session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  setBadge("预热模型…");
  const warm = document.createElement("canvas");
  warm.width = INPUT_SIZE;
  warm.height = INPUT_SIZE;
  warm.getContext("2d").fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  await infer(warm, INPUT_SIZE, INPUT_SIZE, 0.5);

  state.ready = true;
  setBadge(`浏览器端 · WASM${threads > 1 ? " ×" + threads : ""}`);
  els.pickBtn.disabled = false;
  els.camBtn.disabled = false;
  els.statusText.textContent = "等待检测";
  els.statusHint.textContent = "上传图片或开启摄像头开始";
}

/* ------------------------------ inference ------------------------------ */

function letterbox(source, sw, sh) {
  const scale = Math.min(INPUT_SIZE / sw, INPUT_SIZE / sh);
  const w = Math.max(1, Math.round(sw * scale));
  const h = Math.max(1, Math.round(sh * scale));
  const padX = Math.floor((INPUT_SIZE - w) / 2);
  const padY = Math.floor((INPUT_SIZE - h) / 2);

  inputCtx.fillStyle = "#727272";
  inputCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  inputCtx.drawImage(source, padX, padY, w, h);

  const rgba = inputCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const plane = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0, p = 0; i < rgba.length; i += 4, p += 1) {
    inputFloat[p] = rgba[i] / 255;
    inputFloat[p + plane] = rgba[i + 1] / 255;
    inputFloat[p + 2 * plane] = rgba[i + 2] / 255;
  }
  return { scale, padX, padY };
}

async function infer(source, sw, sh, conf) {
  const started = performance.now();
  const meta = letterbox(source, sw, sh);
  const tensor = new ort.Tensor("float32", inputFloat, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const feeds = {};
  feeds[state.session.inputNames[0]] = tensor;
  const output = await state.session.run(feeds);
  const first = output[state.session.outputNames[0]];
  const dets = TLPost.decode(first.data, first.dims, {
    conf,
    iou: 0.45,
    scale: meta.scale,
    padX: meta.padX,
    padY: meta.padY,
    numClasses: 4,
    width: sw,
    height: sh,
  });
  return { dets, ms: performance.now() - started };
}

/* ------------------------------ rendering ------------------------------ */

function roundRect(context, x, y, w, h, r) {
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + w, y, x + w, y + h, r);
  context.arcTo(x + w, y + h, x, y + h, r);
  context.arcTo(x, y + h, x, y, r);
  context.arcTo(x, y, x + w, y, r);
  context.closePath();
}

function drawDetections(dets) {
  if (!dets || !dets.length) return;
  const scaleX = els.canvas.width / (state.frameWidth || els.canvas.width);
  const scaleY = els.canvas.height / (state.frameHeight || els.canvas.height);
  ctx.lineJoin = "round";
  dets.forEach((det) => {
    const [x1, y1, x2, y2] = det.box;
    const X = x1 * scaleX;
    const Y = y1 * scaleY;
    const W = (x2 - x1) * scaleX;
    const H = (y2 - y1) * scaleY;
    const color = COLORS[det.label] || "#ffffff";

    ctx.lineWidth = Math.max(2, Math.round(els.canvas.width / 420));
    ctx.strokeStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.strokeRect(X, Y, W, H);
    ctx.shadowBlur = 0;

    const text = `${det.label_cn || NAMES[det.label] || det.label} ${(det.confidence * 100).toFixed(0)}%`;
    ctx.font = `600 ${Math.max(12, Math.round(els.canvas.width / 62))}px "Segoe UI", "Microsoft YaHei", sans-serif`;
    const padding = 6;
    const tw = ctx.measureText(text).width;
    const th = Math.max(18, Math.round(els.canvas.width / 44));
    let bx = X;
    let by = Y - th - 4;
    if (by < 0) by = Y + 4;
    if (bx + tw + padding * 2 > els.canvas.width) bx = els.canvas.width - tw - padding * 2;
    ctx.fillStyle = color;
    roundRect(ctx, bx, by, tw + padding * 2, th, 5);
    ctx.fill();
    ctx.fillStyle = "#05070a";
    ctx.textBaseline = "middle";
    ctx.fillText(text, bx + padding, by + th / 2 + 1);
  });
}

function renderImage() {
  const img = state.imageCanvas;
  if (!img) return;
  if (els.canvas.width !== img.width || els.canvas.height !== img.height) {
    els.canvas.width = img.width;
    els.canvas.height = img.height;
  }
  ctx.drawImage(img, 0, 0);
  drawDetections(state.dets);
}

function renderLoop() {
  if (!state.running) return;
  const video = els.video;
  if (video.videoWidth) {
    const scale = Math.min(1, 960 / video.videoWidth);
    const width = Math.round(video.videoWidth * scale);
    const height = Math.round(video.videoHeight * scale);
    if (els.canvas.width !== width || els.canvas.height !== height) {
      els.canvas.width = width;
      els.canvas.height = height;
    }
    ctx.drawImage(video, 0, 0, width, height);
    drawDetections(state.dets);
  }
  requestAnimationFrame(renderLoop);
}

/* ------------------------------ results ------------------------------- */

function summarize(dets) {
  const scores = {};
  const counts = {};
  dets.forEach((det) => {
    counts[det.label] = (counts[det.label] || 0) + 1;
    const [x1, y1, x2, y2] = det.box;
    const weight = det.confidence * Math.max(1, (x2 - x1) * (y2 - y1));
    scores[det.label] = (scores[det.label] || 0) + weight;
  });
  const active = Object.entries(scores).filter(([name]) => name !== "off");
  let name = null;
  if (active.length) {
    active.sort((a, b) => b[1] - a[1]);
    name = active[0][0];
  } else if (Object.keys(scores).length) {
    name = Object.keys(scores)[0];
  }
  return { name, counts };
}

function applyResult(dets, ms) {
  state.dets = dets;
  const summary = summarize(dets);
  els.latency.textContent = `${Math.round(ms)} ms`;
  els.count.textContent = String(dets.length);

  if (summary.name) {
    els.statusText.textContent = NAMES[summary.name] || summary.name;
    els.statusText.style.color = COLORS[summary.name] || "#eef2f8";
    els.statusHint.textContent = HINTS[summary.name] || "";
    els.statusCard.style.borderColor = hexToRgba(COLORS[summary.name], 0.35);
    els.statusCard.style.boxShadow = `0 0 34px ${hexToRgba(COLORS[summary.name], 0.12)}`;
  } else {
    els.statusText.textContent = "未检测到信号灯";
    els.statusText.style.color = "#eef2f8";
    els.statusHint.textContent = "换个角度或更清晰的照片试试";
    els.statusCard.style.borderColor = "";
    els.statusCard.style.boxShadow = "";
  }

  const on = { red: false, yellow: false, green: false };
  if (summary.name && on[summary.name] !== undefined) on[summary.name] = true;
  els.lampRed.classList.toggle("on", on.red);
  els.lampYellow.classList.toggle("on", on.yellow);
  els.lampGreen.classList.toggle("on", on.green);

  updateList(dets, summary);
  if (state.mode === "image") renderImage();
}

function updateList(dets, summary) {
  if (!dets.length) {
    els.detList.innerHTML = '<div class="empty">未检测到目标</div>';
    els.summaryCounts.textContent = "";
    return;
  }
  els.detList.innerHTML = dets
    .map((det) => {
      const color = COLORS[det.label] || "#ffffff";
      const pct = Math.round(det.confidence * 100);
      return `<div class="det-item">
        <span class="det-color" style="background:${color};box-shadow:0 0 10px ${hexToRgba(color, 0.6)}"></span>
        <span class="det-name">${det.label_cn || NAMES[det.label] || det.label}<small>置信度</small></span>
        <span class="det-conf">${pct}%</span>
        <span class="det-bar"><i style="width:${pct}%;background:${color}"></i></span>
      </div>`;
    })
    .join("");
  if (summary && summary.counts) {
    els.summaryCounts.textContent = Object.entries(summary.counts)
      .map(([name, n]) => `${NAMES[name] || name} × ${n}`)
      .join(" · ");
  }
}

function hexToRgba(hex, alpha) {
  if (!hex) return "rgba(0,0,0,0)";
  const value = hex.replace("#", "");
  const full = value.length === 3 ? value.split("").map((c) => c + c).join("") : value;
  const bigint = parseInt(full, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/* ------------------------------- events ------------------------------- */

els.conf.addEventListener("input", () => setConf(els.conf.value));
els.conf.addEventListener("change", () => {
  if (state.mode === "image" && state.imageCanvas && state.ready) runOnImage();
});

els.pickBtn.addEventListener("click", () => els.fileInput.click());
els.placeholder.addEventListener("click", () => {
  if (state.ready) els.fileInput.click();
});
els.fileInput.addEventListener("change", (event) => {
  const file = event.target.files && event.target.files[0];
  if (file) handleFile(file);
  els.fileInput.value = "";
});

["dragenter", "dragover"].forEach((name) =>
  els.stage.addEventListener(name, (event) => {
    event.preventDefault();
    if (state.mode === "image") els.placeholder.classList.add("dragging");
  })
);
["dragleave", "drop"].forEach((name) =>
  els.stage.addEventListener(name, (event) => {
    event.preventDefault();
    els.placeholder.classList.remove("dragging");
  })
);
els.stage.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) handleFile(file);
});

els.tabs.forEach((tab) =>
  tab.addEventListener("click", () => {
    const mode = tab.dataset.mode;
    if (mode === state.mode) return;
    els.tabs.forEach((t) => t.classList.toggle("active", t === tab));
    state.mode = mode;
    if (mode === "image") {
      stopCamera();
      els.placeholderText.textContent = "点击或拖拽图片到这里";
      els.hudMode.textContent = "图片模式";
      els.hud.classList.remove("live");
      els.fps.textContent = "静态图片";
      state.dets = [];
      if (state.imageCanvas) {
        renderImage();
        if (state.ready) runOnImage();
      } else {
        showPlaceholder(true);
      }
    } else {
      els.placeholderText.textContent = "正在打开摄像头…";
      els.hudMode.textContent = "实时模式";
      startCamera();
    }
  })
);

els.camBtn.addEventListener("click", () => {
  if (state.running) {
    stopCamera();
    showPlaceholder(true);
    els.placeholderText.textContent = "点击或拖拽图片到这里";
  } else {
    els.tabs.forEach((t) => t.classList.toggle("active", t.dataset.mode === "camera"));
    state.mode = "camera";
    els.hudMode.textContent = "实时模式";
    startCamera();
  }
});

/* ------------------------------- image -------------------------------- */

async function makeScaled(file, maxSide) {
  let source;
  try {
    source = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    source = await createImageBitmap(file);
  }
  const scale = Math.min(1, maxSide / Math.max(source.width, source.height));
  const width = Math.max(1, Math.round(source.width * scale));
  const height = Math.max(1, Math.round(source.height * scale));
  const off = document.createElement("canvas");
  off.width = width;
  off.height = height;
  off.getContext("2d").drawImage(source, 0, 0, width, height);
  if (source.close) source.close();
  return off;
}

async function handleFile(file) {
  if (!state.ready) return;
  if (state.running) stopCamera();
  els.tabs.forEach((t) => t.classList.toggle("active", t.dataset.mode === "image"));
  state.mode = "image";
  els.hudMode.textContent = "图片模式";
  els.hud.classList.remove("live");
  els.fps.textContent = "静态图片";

  try {
    const off = await makeScaled(file, 1280);
    state.imageCanvas = off;
    state.frameWidth = off.width;
    state.frameHeight = off.height;
    state.dets = [];
    showPlaceholder(false);
    renderImage();
    await runOnImage();
  } catch (err) {
    toast("图片读取失败");
  }
}

async function runOnImage() {
  const img = state.imageCanvas;
  if (!img || state.busy || !state.ready) return;
  state.busy = true;
  try {
    const { dets, ms } = await infer(img, img.width, img.height, state.conf);
    applyResult(dets, ms);
  } catch (err) {
    toast("识别失败：" + (err.message || err));
  } finally {
    state.busy = false;
  }
}

/* ------------------------------- camera ------------------------------- */

async function startCamera() {
  if (!state.ready) return;
  if (state.running) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    toast("当前浏览器不支持摄像头，请用 HTTPS 或 localhost 打开");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    state.stream = stream;
    els.video.srcObject = stream;
    await els.video.play();
    state.running = true;
    state.dets = [];
    state.frames = 0;
    state.fpsTimer = performance.now();
    els.camBtn.textContent = "关闭摄像头";
    els.camBtn.classList.add("stop");
    els.hud.classList.add("live");
    showPlaceholder(false);
    requestAnimationFrame(renderLoop);
    detectLoop();
  } catch (err) {
    toast("无法打开摄像头：" + (err.message || err));
    els.camBtn.textContent = "开启摄像头";
    els.camBtn.classList.remove("stop");
    state.mode = "image";
    els.tabs.forEach((t) => t.classList.toggle("active", t.dataset.mode === "image"));
    els.hudMode.textContent = "图片模式";
    els.placeholderText.textContent = "点击或拖拽图片到这里";
    showPlaceholder(true);
  }
}

function stopCamera() {
  state.running = false;
  clearTimeout(state.loopTimer);
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  els.video.srcObject = null;
  els.camBtn.textContent = "开启摄像头";
  els.camBtn.classList.remove("stop");
  els.hud.classList.remove("live");
  els.fps.textContent = "-- FPS";
}

async function detectLoop() {
  if (!state.running) return;
  if (!state.busy && state.ready && els.video.readyState >= 2 && els.video.videoWidth) {
    state.busy = true;
    try {
      const { dets, ms } = await infer(
        els.video,
        els.video.videoWidth,
        els.video.videoHeight,
        state.conf
      );
      state.frameWidth = els.video.videoWidth;
      state.frameHeight = els.video.videoHeight;
      applyResult(dets, ms);
      state.frames += 1;
      const now = performance.now();
      if (now - state.fpsTimer >= 1000) {
        els.fps.textContent = `${state.frames} FPS`;
        state.frames = 0;
        state.fpsTimer = now;
      }
    } catch (err) {
      /* keep the camera running even if one frame fails */
    } finally {
      state.busy = false;
    }
  }
  state.loopTimer = setTimeout(detectLoop, 10);
}

/* -------------------------------- boot -------------------------------- */

setConf(els.conf.value);
showPlaceholder(true);
els.fps.textContent = "静态图片";

initModel().catch((err) => {
  console.error(err);
  setBadge("模型加载失败");
  els.statusText.textContent = "模型加载失败";
  els.statusHint.textContent = err.message || String(err);
  if (location.protocol === "file:") {
    toast("请通过 http:// 打开页面，不能直接双击 HTML 文件");
  } else {
    toast("模型加载失败：" + (err.message || err));
  }
});
