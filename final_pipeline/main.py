"""
main.py
=======
Real-time ANPR pipeline for Raspberry Pi 4 Model B (4 GB).

    Detection : YOLO via onnxruntime          (yolo_detect.py)
    Tracking  : ByteTrack -- OPTIONAL          (tracking.py, settings.USE_BYTETRACK)
    OCR       : RapidOCR via onnxruntime       (rapid_ocr.py)
    Correct   : rule-based plate repair        (rule_based.py)

Detection scheduling is selectable (settings.DETECTION_SCHEDULER):
    video_time  -> once every DETECT_INTERVAL_SEC of VIDEO time (frame_idx/fps);
                   best for recorded files (independent of processing speed).
    wall_clock  -> once every DETECT_INTERVAL_SEC of REAL time; best for a live
                   camera.
    frame       -> once every DETECT_EVERY frames (legacy/debug).
    auto        -> camera => wall_clock, file => video_time.

Two modes (settings.USE_BYTETRACK):
    False (default) : no tracker. Each detect-frame is OCR'd; plates are
                      corrected, then de-duplicated by similarity + voting.
                      Best when DETECT_INTERVAL_SEC is large (~0.5s+) on the Pi.
    True            : per-track-ID OCR caching (best with a small interval).

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

# --- CHALLAN LOOKUP (mock now; real government REST API later) ----------------
# The pipeline knows ONE thing: check_challan(plate). Whether that hits SQLite,
# a mock seeder, or the real government API is entirely hidden behind it.
from mock_government.challan_service import check_challan


# ------------------------------------------------------------------
#  Plate aggregator (used in NO-TRACK mode and for final consolidation)
# ------------------------------------------------------------------
class PlateAggregator:
    """Collects corrected plates, merges look-alikes, counts votes, keeps max area."""

    def __init__(self):
        self.records = {}   # canonical_plate -> {"votes":int, "area":int, "score":float, "valid":bool}
        self.order = []     # canonical plates in first-finalized order (reporting only)

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
                    if plate != ex:                      # keep first-finalized slot, relabel it
                        self.order[self.order.index(ex)] = plate
                else:
                    rec["area"] = max(rec["area"], area)
                return
        self.records[plate] = {"votes": 1, "area": area, "score": score, "valid": valid}
        self.order.append(plate)                          # remember first-finalized position

    def best(self):
        return sorted(self.records.items(),
                      key=lambda kv: (kv[1]["votes"], kv[1]["score"]), reverse=True)


def open_source(path):
    """Accept a video file path or a numeric camera index (as string)."""
    if isinstance(path, str) and path.isdigit():
        return cv2.VideoCapture(int(path))
    return cv2.VideoCapture(path)


def _resolve_scheduler(setting, is_camera, video_fps):
    """
    Turn the DETECTION_SCHEDULER setting into a concrete mode:
    "video_time" | "wall_clock" | "frame".
      - "auto": camera -> wall_clock, file -> video_time.
      - "video_time" needs a valid FPS; if unknown, fall back to wall_clock.
    """
    mode = (setting or "auto").lower()
    if mode == "auto":
        mode = "wall_clock" if is_camera else "video_time"
    if mode == "video_time" and not (video_fps and video_fps > 0):
        mode = "wall_clock"                  # FPS unknown -> real-time is the only sane clock
    if mode not in ("video_time", "wall_clock", "frame"):
        mode = "wall_clock"
    return mode


def _print_scheduler(mode, raw_setting):
    """Startup banner describing the active scheduler."""
    suffix = "  (auto)" if (raw_setting or "").lower() == "auto" else ""
    if mode == "frame":
        print(f"[Main] Detection Scheduler : Frame-Based{suffix}")
        print(f"[Main] Detect Every        : {cfg.DETECT_EVERY} frames")
    elif mode == "video_time":
        print(f"[Main] Detection Scheduler : Video-Time{suffix}")
        print(f"[Main] Interval            : {cfg.DETECT_INTERVAL_SEC:.2f} sec of video")
    else:  # wall_clock
        print(f"[Main] Detection Scheduler : Wall-Clock (real-time){suffix}")
        print(f"[Main] Interval            : {cfg.DETECT_INTERVAL_SEC:.2f} sec real")


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

    aggregator = PlateAggregator()           # no-track + final consolidation
    ocr_cache = {}                           # track_id -> {"text","area"}  (track mode)
    best_crop_area = {}                      # track_id -> area             (track mode)

    total_det_ms = total_ocr_ms = total_loop_ms = 0.0
    ocr_count = frame_idx = detect_count = 0
    last_boxes, last_scores = None, None
    last_draw = []                           # [(xyxy, label)] for display between detections
    last_detect_mark = None                  # last detection's clock value (video_time OR wall_clock)

    cap = open_source(src)
    video_fps = cap.get(cv2.CAP_PROP_FPS)                 # source FPS (for duration calc)
    video_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # total frames in the source

    # Resolve the detection scheduler ("auto" picks per source; video_time needs FPS).
    is_camera = isinstance(src, str) and src.isdigit()
    scheduler = _resolve_scheduler(cfg.DETECTION_SCHEDULER, is_camera, video_fps)
    _print_scheduler(scheduler, cfg.DETECTION_SCHEDULER)
    print("[Main] Ready. Processing...")

    proc_start = time.monotonic()                         # wall-clock start of processing

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        loop_t0 = time.time()

        # --- DETECTION SCHEDULE (video_time / wall_clock / frame) ---
        # Only decides WHEN detection runs. Everything below (tracking/OCR/
        # vote/report) is unchanged and still keyed off `did_detect`.
        if scheduler == "frame":
            # Once every DETECT_EVERY frames (deterministic in frame units).
            did_detect = (frame_idx == 1 or frame_idx % cfg.DETECT_EVERY == 0)
            clock_val = None
        else:
            # "clock" is either VIDEO time (frame_idx/fps) or WALL-CLOCK time.
            clock_val = (frame_idx / video_fps) if scheduler == "video_time" else time.monotonic()
            did_detect = (last_detect_mark is None
                          or (clock_val - last_detect_mark) >= cfg.DETECT_INTERVAL_SEC)

        if did_detect:
            if scheduler != "frame":
                # Advance by one whole interval (stable cadence, no drift)
                # instead of snapping to the clock. If we've fallen more than
                # one interval behind, resynchronize to the current clock value.
                if last_detect_mark is None:
                    last_detect_mark = clock_val
                else:
                    last_detect_mark += cfg.DETECT_INTERVAL_SEC
                    if clock_val - last_detect_mark > cfg.DETECT_INTERVAL_SEC:
                        last_detect_mark = clock_val
            t0 = time.time()
            last_boxes, last_scores = detector.detect(frame)
            total_det_ms += (time.time() - t0) * 1000
            detect_count += 1

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
    processing_sec = time.monotonic() - proc_start        # total wall-clock processing time
    # Real video length = total_frames / source_fps (NOT the effective FPS).
    video_duration_sec = (video_frame_count / video_fps) if video_fps and video_fps > 0 else 0.0

    # In track mode, fold the per-ID cache into the aggregator for consolidation.
    if cfg.USE_BYTETRACK:
        for data in ocr_cache.values():
            if data["text"] not in ("UNKNOWN", "WAITING..."):
                r = correct_plate(data["text"]) if cfg.USE_RULE_BASED else None
                plate = r["plate"] if r else data["text"]
                aggregator.add(plate, data["area"],
                               r["score"] if r else 0.0,
                               r["valid"] if r else False)

    _report(frame_idx, detect_count, total_det_ms, total_ocr_ms, total_loop_ms,
            ocr_count, aggregator, detector.name,
            video_duration_sec, processing_sec, scheduler)


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
def _report(frame_idx, detect_count, total_det_ms, total_ocr_ms, total_loop_ms,
            ocr_count, aggregator, backend, video_duration_sec, processing_sec,
            scheduler):
    avg_det = total_det_ms / detect_count if detect_count else 0
    avg_loop = total_loop_ms / frame_idx if frame_idx else 0
    fps = 1000.0 / avg_loop if avg_loop else 0
    avg_ocr = total_ocr_ms / ocr_count if ocr_count else 0

    print("\n" + "=" * 50)
    print(f"            ANPR BENCHMARK ({backend} + RapidOCR)")
    print("=" * 50)
    print(f"Image size           : {cfg.IMG_SIZE} x {cfg.IMG_SIZE}")
    if scheduler == "frame":
        print(f"Detection Scheduler  : Frame-Based")
        print(f"Detect Every         : {cfg.DETECT_EVERY} frames")
    elif scheduler == "video_time":
        print(f"Detection Scheduler  : Video-Time")
        print(f"Interval             : {cfg.DETECT_INTERVAL_SEC:.2f} sec of video")
    else:
        print(f"Detection Scheduler  : Wall-Clock (real-time)")
        print(f"Interval             : {cfg.DETECT_INTERVAL_SEC:.2f} sec real")
    print(f"ByteTrack            : {'ON' if cfg.USE_BYTETRACK else 'OFF'}")
    print(f"Frames processed     : {frame_idx}")
    print(f"Detections run       : {detect_count}")
    print(f"Frames skipped       : {frame_idx - detect_count}  (no detection)")
    print(f"Avg detect           : {avg_det:.2f} ms (only on detect frames)")
    print(f"Full loop / frame    : {avg_loop:.2f} ms")
    print(f"Effective FPS        : {fps:.2f}")
    print(f"OCR crops processed  : {ocr_count}")
    print(f"Avg OCR / crop       : {avg_ocr:.2f} ms")
    print("-" * 50)
    if video_duration_sec > 0:
        delay = processing_sec - video_duration_sec       # + = slower than real-time
        print(f"Video Duration       : {video_duration_sec:.2f} s")
        print(f"Processing Time      : {processing_sec:.2f} s")
        print(f"Processing Delay     : {delay:+.2f} s")
    else:
        print(f"Video Duration       : N/A (unknown source FPS)")
        print(f"Processing Time      : {processing_sec:.2f} s")
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
    RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
    # Preserve the order plates were FIRST finalized (aggregator.order),
    # NOT alphabetical and NOT vote-ranked. Uniqueness is guaranteed by the dict.
    valid = [p for p in aggregator.order if aggregator.records[p]["valid"]]
    if valid:
        for i, plate in enumerate(valid, 1):
            result = check_challan(plate)      # mock now, real govt API later
            if result["has_challan"]:
                print(f" {i}. {RED}{plate}  -> CHALLAN Rs.{result['amount']}{RESET}")
            else:
                print(f" {i}. {GREEN}{plate}  -> no challan{RESET}")
    else:
        print(" None found.")
    print("=" * 50)


if __name__ == "__main__":
    main()
