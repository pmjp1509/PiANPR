"""
main.py
=======
Real-time ANPR pipeline for Raspberry Pi 4 Model B (4 GB).

    Detection : YOLO via onnxruntime          (yolo_detect.py)
    Tracking  : ByteTrack -- OPTIONAL          (tracking.py, settings.USE_BYTETRACK)
    OCR       : RapidOCR via onnxruntime       (rapid_ocr.py)
    Correct   : rule-based plate repair        (rule_based.py)

Two modes (settings.USE_BYTETRACK):
    False (default) : no tracker. Each detect-frame is OCR'd; plates are
                      corrected, then de-duplicated by similarity + voting.
                      Best when DETECT_EVERY is large (15-30) on the Pi.
    True            : per-track-ID OCR caching (best with small DETECT_EVERY).

Run:
    cd final_pipeline
    python main.py
Press 'q' to quit the preview window.
"""

import os
import time

import cv2

import settings as cfg
from yolo_detect import build_detector
from rapid_ocr import PlateOCR
from rule_based import correct_plate, similar


# ------------------------------------------------------------------
#  Plate aggregator (used in NO-TRACK mode and for final consolidation)
# ------------------------------------------------------------------
class PlateAggregator:
    """Collects corrected plates, merges look-alikes, counts votes, keeps max area."""

    def __init__(self):
        self.records = {}   # canonical_plate -> {"votes":int, "area":int, "score":float, "valid":bool}

    def add(self, plate, area, score, valid):
        if not plate:
            return
        for ex in list(self.records):
            if similar(plate, ex) > cfg.SIMILARITY_THRESHOLD:
                rec = self.records[ex]
                rec["votes"] += 1
                # promote the higher-scoring / larger-area spelling as canonical
                if (score, area) > (rec["score"], rec["area"]):
                    rec.update(area=area, score=score, valid=valid)
                    self.records[plate] = self.records.pop(ex)
                else:
                    rec["area"] = max(rec["area"], area)
                return
        self.records[plate] = {"votes": 1, "area": area, "score": score, "valid": valid}

    def best(self):
        return sorted(self.records.items(),
                      key=lambda kv: (kv[1]["votes"], kv[1]["score"]), reverse=True)


def open_source(path):
    """Accept a video file path or a numeric camera index (as string)."""
    if isinstance(path, str) and path.isdigit():
        return cv2.VideoCapture(int(path))
    return cv2.VideoCapture(path)


def main():
    src = cfg.VIDEO_PATH
    if not (isinstance(src, str) and src.isdigit()) and not os.path.exists(src):
        raise FileNotFoundError(f"Video not found: {src}")

    print(f"[Main] Loading {cfg.DETECTOR_BACKEND.upper()} detector...")
    detector = build_detector()
    print("[Main] Loading RapidOCR (onnxruntime)...")
    ocr = PlateOCR()

    tracker = None
    if cfg.USE_BYTETRACK:
        from tracking import PlateTracker      # import only when enabled
        tracker = PlateTracker()
        print("[Main] ByteTrack: ENABLED")
    else:
        print("[Main] ByteTrack: DISABLED (vote-based de-dup)")
    print(f"[Main] DETECT_EVERY = {cfg.DETECT_EVERY}")
    print("[Main] Ready. Processing...")

    aggregator = PlateAggregator()           # no-track + final consolidation
    ocr_cache = {}                           # track_id -> {"text","area"}  (track mode)
    best_crop_area = {}                      # track_id -> area             (track mode)

    total_det_ms = total_ocr_ms = total_loop_ms = 0.0
    ocr_count = frame_idx = 0
    last_boxes, last_scores = None, None
    last_draw = []                           # [(xyxy, label)] for display between detections

    cap = open_source(src)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        loop_t0 = time.time()

        did_detect = (frame_idx == 1 or frame_idx % cfg.DETECT_EVERY == 0)
        if did_detect:
            t0 = time.time()
            last_boxes, last_scores = detector.detect(frame)
            total_det_ms += (time.time() - t0) * 1000

        if cfg.USE_BYTETRACK:
            last_draw, dt = _run_tracked(frame, tracker, ocr, last_boxes, last_scores,
                                         did_detect, ocr_cache, best_crop_area)
            total_ocr_ms += dt["ms"]; ocr_count += dt["n"]
        elif did_detect:
            last_draw, dt = _run_untracked(frame, ocr, last_boxes, aggregator)
            total_ocr_ms += dt["ms"]; ocr_count += dt["n"]

        # ---- draw current boxes/labels ----
        for (x1, y1, x2, y2), label in last_draw:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if cfg.SHOW_WINDOW:
            disp = frame
            if frame.shape[1] > 1280:
                disp = cv2.resize(frame, (cfg.DISPLAY_WIDTH, cfg.DISPLAY_HEIGHT))
            cv2.imshow(f"ANPR ({detector.name} + RapidOCR)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        total_loop_ms += (time.time() - loop_t0) * 1000

    cap.release()
    cv2.destroyAllWindows()

    # In track mode, fold the per-ID cache into the aggregator for consolidation.
    if cfg.USE_BYTETRACK:
        for data in ocr_cache.values():
            if data["text"] not in ("UNKNOWN", "WAITING..."):
                r = correct_plate(data["text"]) if cfg.USE_RULE_BASED else None
                plate = r["plate"] if r else data["text"]
                aggregator.add(plate, data["area"],
                               r["score"] if r else 0.0,
                               r["valid"] if r else False)

    _report(frame_idx, total_det_ms, total_ocr_ms, total_loop_ms, ocr_count,
            aggregator, detector.name)


def _run_tracked(frame, tracker, ocr, boxes, scores, did_detect,
                 ocr_cache, best_crop_area):
    """ByteTrack path: stable IDs, OCR each ID once (or again if box grew)."""
    draw = []
    ms, n = 0.0, 0
    tracks = tracker.update(boxes, scores)
    for (x1, y1, x2, y2), tid in tracks:
        area = (x2 - x1) * (y2 - y1)
        if area < cfg.MIN_BOX_AREA:
            continue
        if did_detect:
            prev = best_crop_area.get(tid, 0)
            if (tid not in ocr_cache) or (area > prev * 1.3):
                crop = frame[max(0, y1):min(frame.shape[0], y2),
                             max(0, x1):min(frame.shape[1], x2)]
                if crop.size > 0:
                    t0 = time.time()
                    raw = ocr.read(crop)
                    ms += (time.time() - t0) * 1000; n += 1
                    text = _finalize(raw)
                    ocr_cache[tid] = {"text": text, "area": area}
                    best_crop_area[tid] = area
                    print(f"[OCR] Track {tid} -> {text}")
        label = ocr_cache.get(tid, {"text": "WAITING..."})["text"]
        draw.append(((x1, y1, x2, y2), f"ID {tid}: {label}"))
    return draw, {"ms": ms, "n": n}


def _run_untracked(frame, ocr, boxes, aggregator):
    """No-track path: OCR every detection on this detect-frame, aggregate by vote."""
    draw = []
    ms, n = 0.0, 0
    if boxes is None:
        return draw, {"ms": ms, "n": n}
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)
        if area < cfg.MIN_BOX_AREA:
            continue
        crop = frame[max(0, y1):min(frame.shape[0], y2),
                     max(0, x1):min(frame.shape[1], x2)]
        if crop.size == 0:
            continue
        t0 = time.time()
        raw = ocr.read(crop)
        ms += (time.time() - t0) * 1000; n += 1
        if cfg.USE_RULE_BASED:
            r = correct_plate(raw)
            label = r["plate"] if r["plate"] else "UNKNOWN"
            aggregator.add(r["plate"], area, r["score"], r["valid"])
        else:
            label = raw if len(raw) >= cfg.MIN_PLATE_LEN else "UNKNOWN"
            aggregator.add(label, area, 0.0, False)
        print(f"[OCR] {raw or '<empty>'} -> {label}")
        draw.append(((x1, y1, x2, y2), label))
    return draw, {"ms": ms, "n": n}


def _finalize(raw):
    """Apply rule-based correction (track mode label)."""
    if cfg.USE_RULE_BASED:
        r = correct_plate(raw)
        return r["plate"] if r["plate"] else "UNKNOWN"
    return raw if len(raw) >= cfg.MIN_PLATE_LEN else "UNKNOWN"


# ------------------------------------------------------------------
#  Reporting
# ------------------------------------------------------------------
def _report(frame_idx, total_det_ms, total_ocr_ms, total_loop_ms, ocr_count,
            aggregator, backend):
    det_frames = (frame_idx // cfg.DETECT_EVERY) + 1
    avg_det = total_det_ms / det_frames if det_frames else 0
    avg_loop = total_loop_ms / frame_idx if frame_idx else 0
    fps = 1000.0 / avg_loop if avg_loop else 0
    avg_ocr = total_ocr_ms / ocr_count if ocr_count else 0

    print("\n" + "=" * 50)
    print(f"            ANPR BENCHMARK ({backend} + RapidOCR)")
    print("=" * 50)
    print(f"Image size           : {cfg.IMG_SIZE} x {cfg.IMG_SIZE}")
    print(f"DETECT_EVERY         : {cfg.DETECT_EVERY}  (detect runs 1 / {cfg.DETECT_EVERY} frames)")
    print(f"ByteTrack            : {'ON' if cfg.USE_BYTETRACK else 'OFF'}")
    print(f"Frames processed     : {frame_idx}")
    print(f"Avg detect           : {avg_det:.2f} ms (only on detect frames)")
    print(f"Full loop / frame    : {avg_loop:.2f} ms")
    print(f"Effective FPS        : {fps:.2f}")
    print(f"OCR crops processed  : {ocr_count}")
    print(f"Avg OCR / crop       : {avg_ocr:.2f} ms")
    print("=" * 50)

    ranked = aggregator.best()
    print("\n" + "=" * 65)
    print("              PLATE CONSOLIDATION REPORT")
    print("=" * 65)
    if not ranked:
        print(" No plates detected.")
    for plate, rec in ranked:
        tag = "VALID  " if rec["valid"] else "INVALID"
        print(f" {plate:<12} [{tag}] votes:{rec['votes']:<3} "
              f"score:{rec['score']:<4} area:{rec['area']}")
    print("=" * 65)

    print("\n" + "=" * 50)
    print("       FINAL UNIQUE VALID INDIAN PLATES")
    print("=" * 50)
    valid = [p for p, r in ranked if r["valid"]]
    if valid:
        for i, plate in enumerate(sorted(valid), 1):
            print(f" {i}. {plate}")
    else:
        print(" None found.")
    print("=" * 50)


if __name__ == "__main__":
    main()
