"""
yolo_detect.py
==============
YOLO detection. Two interchangeable backends:

    OnnxDetector  -- onnxruntime (CPUExecutionProvider). Default; fastest here.
    NcnnDetector  -- Tencent ncnn. Works, but measured slower than ONNX on this
                     Pi, so it is opt-in (settings.DETECTOR_BACKEND = "ncnn").

Both return plain numpy arrays (NO supervision dependency); tracking is layered
on top separately:
    boxes  : (N, 4) float32  xyxy in original-frame pixels
    scores : (N,)  float32   confidences

Use build_detector() to get the one configured in settings.
"""

import cv2
import numpy as np
import onnxruntime as ort

import settings as cfg


def letterbox(img, new_shape=cfg.IMG_SIZE, color=(114, 114, 114)):
    """Resize keeping aspect ratio, pad to a square. Returns (canvas, ratio, pad_w, pad_h)."""
    h0, w0 = img.shape[:2]
    r = min(new_shape / h0, new_shape / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    pad_w = (new_shape - new_w) // 2
    pad_h = (new_shape - new_h) // 2
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return canvas, r, pad_w, pad_h


def _empty():
    return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)


def _postprocess(arr, w0, h0, r, pad_w, pad_h):
    """
    Decode raw YOLOv8 output (shape (8400, 5) for single class), undo letterbox,
    run NMS. Returns (boxes_xyxy, scores) in original-frame pixels.
    """
    scores = arr[:, 4]
    mask = scores >= cfg.CONF_THRESHOLD
    if not mask.any():
        return _empty()

    box = arr[mask][:, :4]
    scores = scores[mask]
    cx, cy, w, h = box.T
    x1 = (cx - w / 2 - pad_w) / r
    y1 = (cy - h / 2 - pad_h) / r
    x2 = (cx + w / 2 - pad_w) / r
    y2 = (cy + h / 2 - pad_h) / r
    x1 = np.clip(x1, 0, w0); x2 = np.clip(x2, 0, w0)
    y1 = np.clip(y1, 0, h0); y2 = np.clip(y2, 0, h0)

    nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    idxs = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(),
                            cfg.CONF_THRESHOLD, cfg.IOU_THRESHOLD)
    if len(idxs) == 0:
        return _empty()
    idxs = np.array(idxs).flatten()

    boxes = np.stack([x1[idxs], y1[idxs], x2[idxs], y2[idxs]], axis=1).astype(np.float32)
    conf = scores[idxs].astype(np.float32)
    return boxes, conf


class OnnxDetector:
    name = "ONNX"

    def __init__(self, model_path=cfg.MODEL_PATH):
        so = ort.SessionOptions()
        so.intra_op_num_threads = cfg.NUM_THREADS          # use all Pi cores
        # ORT_ENABLE_ALL fuses nodes -> faster inference (slower session build).
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame):
        h0, w0 = frame.shape[:2]
        canvas, r, pad_w, pad_h = letterbox(frame)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        inp = rgb.astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None]

        out = self.session.run(None, {self.input_name: inp})[0]
        arr = np.squeeze(out, 0).T            # (8400, 5) for single-class YOLOv8
        return _postprocess(arr, w0, h0, r, pad_w, pad_h)


class NcnnDetector:
    name = "NCNN"

    def __init__(self, param_path=cfg.NCNN_PARAM_PATH, bin_path=cfg.NCNN_BIN_PATH):
        import ncnn                            # imported only when NCNN is selected
        self._ncnn = ncnn
        self.net = ncnn.Net()
        self.net.opt.num_threads = cfg.NCNN_THREADS
        self.net.load_param(param_path)
        self.net.load_model(bin_path)

    def detect(self, frame):
        ncnn = self._ncnn
        h0, w0 = frame.shape[:2]
        canvas, r, pad_w, pad_h = letterbox(frame)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        mat = ncnn.Mat.from_pixels(rgb, ncnn.Mat.PixelType.PIXEL_RGB,
                                   cfg.IMG_SIZE, cfg.IMG_SIZE)
        mat.substract_mean_normalize([0., 0., 0.], [1 / 255., 1 / 255., 1 / 255.])

        ex = self.net.create_extractor()
        ex.input("in0", mat)
        _, out = ex.extract("out0")
        arr = np.array(out).T                  # (8400, 5)
        return _postprocess(arr, w0, h0, r, pad_w, pad_h)


def build_detector():
    """Construct the detector chosen in settings.DETECTOR_BACKEND."""
    backend = cfg.DETECTOR_BACKEND.lower()
    if backend == "ncnn":
        return NcnnDetector()
    if backend == "onnx":
        return OnnxDetector()
    raise ValueError(f"Unknown DETECTOR_BACKEND: {cfg.DETECTOR_BACKEND!r} (use 'onnx' or 'ncnn')")
