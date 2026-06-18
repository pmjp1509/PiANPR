# ANPR Pipeline (Raspberry Pi 4 Model B, 4 GB)

Automatic Number Plate Recognition for Indian plates, structured into small,
single-responsibility modules. Built to actually run on a Pi 4: detection is
sampled (not every frame), tracking is optional, and OCR output is repaired
with rule-based correction.

```
Detection : YOLO via onnxruntime          (yolo_detect.py)
Tracking  : ByteTrack -- OPTIONAL          (tracking.py)
OCR       : RapidOCR via onnxruntime       (rapid_ocr.py)   # no PaddlePaddle
Correct   : rule-based plate repair        (rule_based.py)
Config    : all thresholds / paths         (settings.py)
Orchestr. : the loop + reporting           (main.py)
```

## Why this stack

PaddlePaddle's aarch64 build segfaults during inference on the Pi. RapidOCR runs
the **same** PP-OCR recognition models on `onnxruntime`, which already works on
the Pi. So YOLO and OCR share one runtime and there is no Paddle dependency.

## Install

```bash
cd final_pipeline
pip install -r requirements.txt
# reclaim space (these segfault on the Pi anyway):
pip uninstall paddlepaddle paddlex paddleocr
```

First OCR run downloads ~15 MB of models (needs internet once).

## Run

```bash
python main.py        # press 'q' to quit the preview
```

Edit **`settings.py`** for everything — model/video paths, thresholds, and the
two big switches below. Set `VIDEO_PATH = "0"` to use a USB/Pi camera.

## The switches you care about

### `DETECTOR_BACKEND` (default `"onnx"`)
`"onnx"` (onnxruntime) or `"ncnn"` (Tencent ncnn). Both are implemented in
`yolo_detect.py` and return identical output; the active one is printed at
startup, in the preview title, and in the benchmark header. NCNN works but
measured **slower** than ONNX on this Pi, so ONNX is the default. To try NCNN:
`yolo export model=best.pt format=ncnn`, point `NCNN_PARAM_PATH` / `NCNN_BIN_PATH`
at the result, set `DETECTOR_BACKEND = "ncnn"`, and `pip install ncnn`.

### `DETECT_EVERY` (default 15)
YOLO is ~550 ms and OCR ~250 ms on the Pi, so running detection every frame is
hopeless. We run it once every `DETECT_EVERY` frames and reuse the boxes in
between. 15–30 keeps the loop responsive. The active value is printed at startup
and in the benchmark.

### `USE_BYTETRACK` (default `False`)
ByteTrack gives each vehicle a stable ID so a plate is OCR'd once. **But it
assumes detection runs often enough that boxes overlap between frames.** With
`DETECT_EVERY = 15–30`, fast-moving vehicles jump too far and ByteTrack assigns
a *new* ID anyway — so it costs CPU for no benefit. Hence it is **off by
default**.

| Mode | Behaviour | Use when |
|------|-----------|----------|
| `USE_BYTETRACK = False` | OCR every detect-frame; de-dup by text similarity + voting | sparse detection (`DETECT_EVERY` ≥ 15) — **recommended on the Pi** |
| `USE_BYTETRACK = True`  | per-ID OCR caching | dense detection (`DETECT_EVERY` ≈ 3–5) |

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

## Output

At the end you get a benchmark (with `DETECT_EVERY` and ByteTrack state printed
first), a consolidation report (every plate with votes/score/validity), and the
final list of unique **valid** Indian plates.

---

## Making it faster on the Pi 4


1. **Smaller detector input.** In `settings.py` set `IMG_SIZE = 416` or `320`
   and re-export your model at that size. Inference cost scales ~quadratically
   with input side — 640→320 is up to ~4× faster, with some recall loss on small
   plates. Biggest single win.
2. **Switch the detector to NCNN.** NCNN is ARM-NEON optimised and usually beats
   onnxruntime-CPU for YOLO on the Pi. 
   [Qengineering's YoloV8-ncnn-Raspberry-Pi-4](https://github.com/Qengineering/YoloV8-ncnn-Raspberry-Pi-4).
3. **INT8 quantise the ONNX model.** Dynamic/static INT8 can roughly **2×** the
   speed on the Pi 4 via onnxruntime's
   [quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
   Test accuracy after — INT8 can drop recall on small text/plates.
4. **Keep `graph_optimization_level = ORT_ENABLE_ALL`** (already set) and
   `intra_op_num_threads = 4` (`NUM_THREADS`). If you see latency *spikes*, try
   2–3 threads — fewer threads can reduce scheduling jitter on the Pi.
5. **Increase `DETECT_EVERY`.** Detection dominates; sampling less often is the
   cheapest knob. Pair with `USE_BYTETRACK = False`.
6. **Threaded capture / detection.** Run frame capture (and ideally detection)
   on a separate thread/process so OCR and display don't stall the grabber. hides latency well on 4 cores.
7. **Crop OCR tightly & OCR fewer crops.** Skip boxes below `MIN_BOX_AREA`, and
   only OCR a plate again if the box grew (already done in track mode). RapidOCR
   recognition-only (no det/cls — already set) is the fast path.
8. **OS-level:** 64-bit Raspberry Pi OS (aarch64), active cooling so the CPU
   doesn't thermal-throttle, and `opencv-python-headless` + `SHOW_WINDOW=False`
   when running over SSH to save the display/GUI overhead.

A realistic target after #1+#2+#6: detection in the ~100–200 ms range, giving a
usable near-real-time loop for a single camera.

### Sources
- [Qengineering — YoloV8 NCNN for Raspberry Pi 4/5](https://github.com/Qengineering/YoloV8-ncnn-Raspberry-Pi-4)
- [onnxruntime — Quantize ONNX models](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [Ultralytics — speeding up YOLO on Raspberry Pi](https://github.com/ultralytics/ultralytics/issues/21167)
- [PyTorch — Real-time inference on Raspberry Pi 4](https://docs.pytorch.org/tutorials/intermediate/realtime_rpi.html)
