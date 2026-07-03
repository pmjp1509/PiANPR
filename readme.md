# ANPR Pipeline for Edge Devices

Automatic Number Plate Recognition for Indian plates, structured into small,
single-responsibility modules. Designed for deployment on resource-constrained
edge devices: detection is sampled (not every frame), tracking is optional, and
OCR output is repaired with rule-based correction.

The software is hardware-independent and portable. It is developed and
benchmarked on a Raspberry Pi 4 Model B (4 GB) as the reference platform, but
nothing is Pi-specific — it runs anywhere onnxruntime and OpenCV are available.

```
final_pipeline/
  main.py           loop, scheduling, benchmark, reporting
  yolo_detect.py    YOLO detection (onnxruntime / ncnn backends)
  rapid_ocr.py      RapidOCR recognition (onnxruntime, no PaddlePaddle)
  tracking.py       ByteTrack wrapper (optional)
  rule_based.py     Indian-plate OCR correction
  settings.py       all config (paths, thresholds, switches)
  mock_government/  mock challan DB, stands in for a real govt API
```

Flow: `frame → YOLO → (ByteTrack) → RapidOCR → rule-based fix → vote/dedup → challan check → report`.

## Why this stack

PaddlePaddle's aarch64 build segfaults during inference on ARM edge devices
(including the Raspberry Pi). RapidOCR runs the **same** PP-OCR recognition
models on `onnxruntime`, which works reliably there. So YOLO and OCR share one
runtime and there is no Paddle dependency.

## Install

```bash
cd final_pipeline
pip install -r requirements.txt
# reclaim space (these segfault on ARM edge devices anyway):
pip uninstall paddlepaddle paddlex paddleocr
```

First OCR run downloads ~15 MB of models (needs internet once).

## Run

```bash
python main.py        # press 'q' to quit the preview
```

Edit **`settings.py`** for everything — model/video paths, thresholds, and the
switches below. Set `VIDEO_PATH = "0"` to use a USB or CSI camera (full list in
[Settings reference](#settings-reference)).

## The switches you care about

### `DETECTOR_BACKEND` (default `"onnx"`)
`"onnx"` (onnxruntime) or `"ncnn"` (Tencent ncnn). Both are implemented in
`yolo_detect.py` and return identical output; the active one is printed at
startup, in the preview title, and in the benchmark header. NCNN works but
measured **slower** than ONNX on our test device (Pi 4), so ONNX is the default.
To try NCNN:
`yolo export model=best.pt format=ncnn`, point `NCNN_PARAM_PATH` / `NCNN_BIN_PATH`
at the result, set `DETECTOR_BACKEND = "ncnn"`, and `pip install ncnn`.

### `DETECTION_SCHEDULER` (default `"auto"`)
On a constrained edge device YOLO + OCR is too slow to run every frame (our test
device, a Pi 4: ~550 ms + ~250 ms), so detection runs periodically and the boxes
are reused in between. This setting picks *how* "periodically" is measured:

- `video_time` — every `DETECT_INTERVAL_SEC` of **video** time (`frame_idx / fps`).
  Cadence is fixed to the video, not the device's processing speed. Best for recorded files.
- `wall_clock` — every `DETECT_INTERVAL_SEC` of **real** time. Best for a live camera.
- `frame` — every `DETECT_EVERY` frames. Legacy / debug.
- `auto` — camera → `wall_clock`, file → `video_time`. Default.

The resolved mode is printed at startup and in the benchmark.

### `USE_BYTETRACK` (default `False`)
ByteTrack gives each vehicle a stable ID so a plate is OCR'd once. **But it
assumes detection runs often enough that boxes overlap between frames.** With a
large `DETECT_INTERVAL_SEC` (~0.5s+), fast-moving vehicles jump too far and
ByteTrack assigns a *new* ID anyway — so it costs CPU for no benefit. Hence it
is **off by default**.

| Mode | Behaviour | Use when |
|------|-----------|----------|
| `USE_BYTETRACK = False` | OCR every detect-frame; de-dup by text similarity + voting | sparse detection (interval ≥ 0.5s) — **recommended on edge devices** |
| `USE_BYTETRACK = True`  | per-ID OCR caching | dense detection (small interval) |

In no-track mode the same plate is seen on several detect-frames; votes make the
final result robust even without IDs.

## Rule-based plate correction (`rule_based.py`)

OCR confuses look-alikes (`6↔G`, `H↔M`, `0↔O`, `1↔I`, `5↔S`, `8↔B`). A regex can
only accept/reject — it can't *repair*. Instead we:

1. **Normalise** — uppercase, strip spaces/hyphens/symbols.
2. **Try each format** for the given length:
   - Standard `SS DD L[1..3] NNNN` (len 9–11)
   - BH-series `YY BH NNNN L[1..2]` (len 9–10)
3. **Position-aware correction** — where a letter is expected, map stray digits
   → letters; where a digit is expected, map stray letters → digits.
4. **State-code repair** — snap the first two letters to the nearest valid
   state/UT (confusion-aware substitution, then edit-distance ≤ 1). e.g.
   `TM → TN`, `6J → GJ`.
5. **District (RTO) check** — light penalty for a 2-digit code above the known
   max for that state (`RTO_MAX`, partial table; never rejects).
6. **Score candidates** by total correction "penalty" and return the valid one
   needing the fewest, most likely fixes.

Examples (`python rule_based.py`):

```
TN1OABI234  -> TN10AB1234  fixes the O→0, I→1
TM10AB1234  -> TN10AB1234  fixes the state code
22BH1234AB  -> 22BH1234AB  recognised BH-series
HR26DQ5551  -> HR26DQ5551  valid as-is
```

`RTO_MAX` is partial — extend it from the
[RTO district list](https://en.wikipedia.org/wiki/List_of_Regional_Transport_Office_districts_in_India)
for the states you operate in (or swap the `int(dd) > max` check for an explicit
per-state set of valid codes if you need exact validation).

## Mock government challan service (`mock_government/`)

Stands in for a real government challan API so the demo runs offline. The
pipeline only ever calls one function:

```python
from mock_government.challan_service import check_challan
check_challan("TN11A6701")   # -> {"plate": "TN11A6701", "has_challan": True, "amount": 1500}
```

- `database.py` — SQLite table (`plate`, `total_challan`) with a few known plates.
- `challan_service.py` — the single interface. Swap in the real API here later.
- `testing_seed.py` — dev-only: gives unseen plates a random amount (70% none,
  30% ₹500–5000) so the DB fills up while testing.

Seeding is controlled by `ENABLE_TESTING` in `challan_service.py`. Set it `False`
(and delete `testing_seed.py`) for production — `main.py` never imports the seeder.

## Output

Three blocks are printed when the run finishes:

- **Benchmark** — backend, image size, scheduler + interval, frames processed /
  detections run / skipped, avg detect & OCR ms, effective FPS, and video
  duration vs. processing time (with the +/- delay).
- **Consolidation report** — every plate seen, with votes / score / valid flag.
- **Final unique valid plates** — in the order they were first finalised, each
  checked against the challan DB: **red** = has a challan, **green** = clean.

## Settings reference

Everything lives in `settings.py`.

**Paths**
- `MODEL_PATH` — path to the `.onnx` detector weights.
- `VIDEO_PATH` — video file path, or `"0"` for a camera.

**Detector**
- `DETECTOR_BACKEND` — `"onnx"` or `"ncnn"`.
- `IMG_SIZE` — detector input size (640 / 512 / 416 / 320). Smaller = faster, weaker on small plates.
- `CONF_THRESHOLD` — min detection confidence. Lower to catch more, raise to cut false boxes.
- `IOU_THRESHOLD` — NMS overlap threshold.
- `MIN_BOX_AREA` — ignore plate boxes smaller than this (drops far-away noise).
- `NUM_THREADS` — onnxruntime CPU threads (match the device core count, e.g. 4 on a Pi 4).
- `NCNN_PARAM_PATH` / `NCNN_BIN_PATH` / `NCNN_THREADS` — only used with the ncnn backend.

**Detection scheduler**
- `DETECTION_SCHEDULER` — `auto` / `video_time` / `wall_clock` / `frame`.
- `DETECT_INTERVAL_SEC` — seconds between detections (`video_time`, `wall_clock`). Raise for more speed.
- `DETECT_EVERY` — frames between detections (`frame` mode only).

**Tracking**
- `USE_BYTETRACK` — enable per-vehicle ID tracking. Off by default.
- `BYTETRACK_*` — tracker tuning; only used when `USE_BYTETRACK = True`.

**OCR / correction**
- `MIN_PLATE_LEN` — ignore OCR strings shorter than this.
- `USE_RULE_BASED` — enable the OCR correction in `rule_based.py`.
- `SIMILARITY_THRESHOLD` — how similar two reads must be to merge as one plate (0–1).
- `VALIDATE_DISTRICT` — penalise unknown RTO district codes.

**Display**
- `SHOW_WINDOW` — show the preview window. Set `False` when headless / over SSH.
- `DISPLAY_WIDTH` / `DISPLAY_HEIGHT` — preview size for large frames.

---

## Making it faster on edge devices


1. **Smaller detector input.** In `settings.py` set `IMG_SIZE = 416` or `320`
   and re-export your model at that size. Inference cost scales ~quadratically
   with input side — 640→320 is up to ~4× faster, with some recall loss on small
   plates. Biggest single win.
2. **Switch the detector to NCNN.** NCNN is ARM-NEON optimised and usually beats
   onnxruntime-CPU for YOLO on ARM edge devices.
   [Qengineering's YoloV8-ncnn-Raspberry-Pi-4](https://github.com/Qengineering/YoloV8-ncnn-Raspberry-Pi-4).
3. **INT8 quantise the ONNX model.** Dynamic/static INT8 can roughly **2×** the
   speed on ARM edge devices via onnxruntime's
   [quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
   Test accuracy after — INT8 can drop recall on small text/plates.
4. **Keep `graph_optimization_level = ORT_ENABLE_ALL`** (already set) and
   `intra_op_num_threads = 4` (`NUM_THREADS`). If you see latency *spikes*, try
   2–3 threads — fewer threads can reduce scheduling jitter on constrained devices.
5. **Increase `DETECT_INTERVAL_SEC`.** Detection dominates; sampling less often
   (a longer interval) is the cheapest knob. Pair with `USE_BYTETRACK = False`.
6. **Threaded capture / detection.** Run frame capture (and ideally detection)
   on a separate thread/process so OCR and display don't stall the grabber —
   hides latency well on 4 cores.
7. **Crop OCR tightly & OCR fewer crops.** Skip boxes below `MIN_BOX_AREA`, and
   only OCR a plate again if the box grew (already done in track mode). RapidOCR
   recognition-only (no det/cls — already set) is the fast path.
8. **OS-level:** a 64-bit OS (aarch64; e.g. Raspberry Pi OS 64-bit), active
   cooling so the CPU doesn't thermal-throttle, and `opencv-python-headless` +
   `SHOW_WINDOW=False` when running headless/over SSH to save display overhead.

A realistic target after #1+#2+#6: detection in the ~100–200 ms range, giving a
usable near-real-time loop for a single camera.

### Sources
- [Qengineering — YoloV8 NCNN for Raspberry Pi 4/5](https://github.com/Qengineering/YoloV8-ncnn-Raspberry-Pi-4)
- [onnxruntime — Quantize ONNX models](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [Ultralytics — speeding up YOLO on Raspberry Pi](https://github.com/ultralytics/ultralytics/issues/21167)
- [PyTorch — Real-time inference on Raspberry Pi 4](https://docs.pytorch.org/tutorials/intermediate/realtime_rpi.html)
