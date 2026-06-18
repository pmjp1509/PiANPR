"""
settings.py
===========
Central configuration for the ANPR pipeline. Edit values here only -- no other
module hard-codes paths or thresholds.

Tuned for a Raspberry Pi 4 Model B (4 GB). On the Pi, YOLO ~550 ms and OCR
~250 ms per call, so detection is run only once every DETECT_EVERY frames.
"""

import os

# ------------------------------------------------------------------
#  Paths
# ------------------------------------------------------------------
# Resolve relative to this file so the pipeline runs from any working dir.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Detector weights. Point this at your .onnx (CPUExecutionProvider) model.
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best.onnx")

# Input source:
#   - a video file path, e.g. "/home/prakash/Desktop/anpr/datasets/clip.mp4"
#   - an integer (as a string) for a camera, e.g. "0" for /dev/video0
VIDEO_PATH = os.path.join(BASE_DIR, "..", "datasets", "croped_vechicle.mp4")

# ------------------------------------------------------------------
#  Detector (YOLO)
# ------------------------------------------------------------------
# Which backend to run detection with:
#   "onnx" -> onnxruntime (DEFAULT; fastest on this Pi)
#   "ncnn" -> Tencent ncnn (works, but measured slower than ONNX here)
DETECTOR_BACKEND = "onnx"

IMG_SIZE        = 640      # network input (square). Try 416/320 on the Pi for speed.
CONF_THRESHOLD  = 0.15
IOU_THRESHOLD   = 0.45
MIN_BOX_AREA    = 1000     # ignore tiny boxes (far-away / spurious plates)
NUM_THREADS     = 4        # Pi 4 has 4 cores; onnxruntime intra-op threads

# NCNN backend files (only used when DETECTOR_BACKEND = "ncnn").
# Export with:  yolo export model=best.pt format=ncnn
NCNN_PARAM_PATH = os.path.join(BASE_DIR, "..", "models", "best_ncnn_model", "model.ncnn.param")
NCNN_BIN_PATH   = os.path.join(BASE_DIR, "..", "models", "best_ncnn_model", "model.ncnn.bin")
NCNN_THREADS    = 4        # Pi 4 has 4 cores

# ------------------------------------------------------------------
#  Detection cadence
# ------------------------------------------------------------------
# Run the detector once every N frames. Higher = faster but stale boxes.
# On the Pi, 15-30 keeps the loop responsive. With ByteTrack OFF this only
# affects how often new plates are sampled (see USE_BYTETRACK).
DETECT_EVERY = 15

# ------------------------------------------------------------------
#  Tracking (OPTIONAL)
# ------------------------------------------------------------------
# ByteTrack keeps a stable ID per vehicle so each plate is OCR'd once.
# BUT it assumes detection runs often enough that boxes overlap frame-to-frame.
# When DETECT_EVERY is large (15-30) on the Pi, vehicles move too far between
# detections and ByteTrack assigns new IDs anyway -- so it adds cost for no gain.
#
#   USE_BYTETRACK = False  ->  no tracker. Every detect-frame is OCR'd and
#                              results are de-duplicated by plate text + voting.
#                              RECOMMENDED for sparse detection on the Pi.
#   USE_BYTETRACK = True   ->  per-ID OCR caching (good only if DETECT_EVERY is
#                              small, ~3-5, so tracks stay alive).
USE_BYTETRACK = False

# ByteTrack params (only used when USE_BYTETRACK = True)
BYTETRACK_ACTIVATION_THRESHOLD = 0.20
BYTETRACK_LOST_BUFFER          = 90
BYTETRACK_MATCH_THRESHOLD      = 0.90
BYTETRACK_FRAME_RATE           = 30

# ------------------------------------------------------------------
#  OCR (RapidOCR -> onnxruntime, no PaddlePaddle)
# ------------------------------------------------------------------
MIN_PLATE_LEN = 4          # ignore OCR strings shorter than this

# ------------------------------------------------------------------
#  Rule-based correction (see rule_based.py)
# ------------------------------------------------------------------
USE_RULE_BASED      = True   # correct OCR with state/format rules
SIMILARITY_THRESHOLD = 0.80  # merge plates this similar when de-duplicating
# Validate the 2-digit district code against a per-state RTO whitelist.
# Our RTO table is partial, so this only *penalises* unknown codes for states
# we have data for; it never rejects outright. See rule_based.RTO_MAX.
VALIDATE_DISTRICT   = True

# ------------------------------------------------------------------
#  Display
# ------------------------------------------------------------------
SHOW_WINDOW   = True       # set False when running headless over SSH
DISPLAY_WIDTH = 854        # downscale large frames for the preview window
DISPLAY_HEIGHT = 480
