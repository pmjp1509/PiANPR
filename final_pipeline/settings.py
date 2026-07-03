"""
settings.py
===========
Central configuration for the ANPR pipeline. Edit values here only -- no other
module hard-codes paths or thresholds.

Tuned for a Raspberry Pi 4 Model B (4 GB). On the Pi, YOLO ~550 ms and OCR
~250 ms per call, so detection is run on a fixed TIME interval
(DETECT_INTERVAL_SEC) rather than every N frames -- this keeps the real-world
detection rate constant across cameras/videos of different FPS.
"""

import os

# ------------------------------------------------------------------
#  Paths
# ------------------------------------------------------------------
# Resolve relative to this file so the pipeline runs from any working dir.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Detector weights. Point this at your .onnx (CPUExecutionProvider) model.
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best1.onnx")

# Input source:
#   - a video file path, e.g. "/home/prakash/Desktop/anpr/datasets/clip.mp4"
#   - an integer (as a string) for a camera, e.g. "0" for /dev/video0
VIDEO_PATH = os.path.join(BASE_DIR, "..", "datasets", "pexels-casey-whalen-6571483 (2160p).mp4")
# ------------------------------------------------------------------
#  Detector (YOLO)
# ------------------------------------------------------------------
# Which backend to run detection with:
#   "onnx" -> onnxruntime (DEFAULT, dynamic img_size, CPUExecutionProvider)
#   "ncnn" -> Tencent ncnn (img_size 640)
DETECTOR_BACKEND = "onnx"

IMG_SIZE        = 512      # network input (square). Try 512/416/320 on the Pi for speed. 640 for ncnn.
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
#  Detection cadence  (choose ONE scheduler)
# ------------------------------------------------------------------
# There are THREE clocks you can schedule detection against, because "detect
# every 1 second" is ambiguous:
#
#   "video_time"  -> every DETECT_INTERVAL_SEC of VIDEO time (frame_idx / fps).
#                    "1 second of video" regardless of how fast/slow the Pi
#                    processes it -- deterministic and tied to video content.
#                    CORRECT for recorded VIDEO FILES.
#
#   "wall_clock"  -> every DETECT_INTERVAL_SEC of REAL time (time.monotonic()).
#                    "1 second of real time". CORRECT for a LIVE CAMERA, where
#                    wall-clock == capture time.
#
#   "frame"       -> every DETECT_EVERY frames (legacy / debug). Deterministic
#                    in frame units; the original frame-based behaviour.
#
#   "auto"        -> pick automatically: a numeric camera source uses
#                    "wall_clock", a video file uses "video_time". If the file's
#                    FPS is unknown, "video_time" safely falls back to
#                    "wall_clock". Recommended -- you don't have to remember.
#
# Why this matters: reading a file, cap.read() is consumed as fast as the CPU
# allows (NOT paced to the file's FPS), so wall-clock seconds do NOT map to a
# fixed number of video frames -- which is why a wall-clock interval felt less
# smooth than the old frame cadence. video_time restores the intended meaning.
DETECTION_SCHEDULER = "auto"    # "auto" | "video_time" | "wall_clock" | "frame"

DETECT_INTERVAL_SEC = 0.5    # used by "video_time" and "wall_clock"
DETECT_EVERY        = 15     # used by "frame"

# ------------------------------------------------------------------
#  Tracking (OPTIONAL)
# ------------------------------------------------------------------
# ByteTrack keeps a stable ID per vehicle so each plate is OCR'd once.
# BUT it assumes detection runs often enough that boxes overlap frame-to-frame.
# When DETECT_INTERVAL_SEC is large (~0.5s+) on the Pi, vehicles move too far
# between detections and ByteTrack assigns new IDs anyway -- so it adds cost for
# no gain.
#
#   USE_BYTETRACK = False  ->  no tracker. Every detect-frame is OCR'd and
#                              results are de-duplicated by plate text + voting.
#                              RECOMMENDED for sparse detection on the Pi.
#   USE_BYTETRACK = True   ->  per-ID OCR caching (good only if
#                              DETECT_INTERVAL_SEC is small, so tracks stay alive).
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
