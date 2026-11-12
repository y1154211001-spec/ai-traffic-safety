(function (root) {
  const CLASS_NAMES = ["traffic_light"];

  function iou(a, b) {
    const x1 = Math.max(a[0], b[0]);
    const y1 = Math.max(a[1], b[1]);
    const x2 = Math.min(a[2], b[2]);
    const y2 = Math.min(a[3], b[3]);
    const w = Math.max(0, x2 - x1);
    const h = Math.max(0, y2 - y1);
    const inter = w * h;
    const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
    const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
    return inter / (areaA + areaB - inter + 1e-9);
  }

  function nms(boxes, iouThreshold) {
    boxes.sort((p, q) => q.conf - p.conf);
    const keep = [];
    const suppressed = new Array(boxes.length).fill(false);
    for (let i = 0; i < boxes.length; i++) {
      if (suppressed[i]) continue;
      keep.push(boxes[i]);
      for (let j = i + 1; j < boxes.length; j++) {
        if (suppressed[j] || boxes[j].cls !== boxes[i].cls) continue;
        if (iou(boxes[i].box, boxes[j].box) > iouThreshold) suppressed[j] = true;
      }
    }
    return keep;
  }

  function decode(data, dims, opts) {
    const numClasses = opts.numClasses || 1;
    const channelsFirst = dims.length === 3 && dims[1] < dims[2];
    const C = channelsFirst ? dims[1] : dims[2];
    const N = channelsFirst ? dims[2] : dims[1];
    const at = channelsFirst
      ? (c, i) => data[c * N + i]
      : (c, i) => data[i * C + c];

    const candidates = [];
    for (let i = 0; i < N; i++) {
      let bestClass = -1;
      let bestScore = 0;
      for (let c = 0; c < numClasses; c++) {
        const score = at(4 + c, i);
        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
        }
      }
      if (bestScore < opts.conf) continue;
      const cx = at(0, i);
      const cy = at(1, i);
      const w = at(2, i);
      const h = at(3, i);
      candidates.push({
        cls: bestClass,
        conf: bestScore,
        box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
      });
    }

    const kept = nms(candidates, opts.iou == null ? 0.45 : opts.iou);
    const scale = opts.scale || 1;
    const padX = opts.padX || 0;
    const padY = opts.padY || 0;
    const maxW = opts.width || Infinity;
    const maxH = opts.height || Infinity;

    return kept.slice(0, 100).map((det) => ({
      label: CLASS_NAMES[det.cls] || String(det.cls),
      confidence: Math.round(det.conf * 10000) / 10000,
      box: [
        Math.min(maxW, Math.max(0, (det.box[0] - padX) / scale)),
        Math.min(maxH, Math.max(0, (det.box[1] - padY) / scale)),
        Math.min(maxW, Math.max(0, (det.box[2] - padX) / scale)),
        Math.min(maxH, Math.max(0, (det.box[3] - padY) / scale)),
      ],
    }));
  }

  root.TLPost = { decode, iou, nms, CLASS_NAMES };
  if (typeof module !== "undefined" && module.exports) module.exports = root.TLPost;
})(typeof window !== "undefined" ? window : globalThis);
