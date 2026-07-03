"""
tracking.py
===========
Optional ByteTrack wrapper (supervision). Only imported by main.py when
settings.USE_BYTETRACK is True, so a Pi running tracker-less doesn't pay the
supervision import cost.

See settings.USE_BYTETRACK for *why* tracking is optional: with sparse
detection (DETECT_EVERY >= 15) vehicles move too far between detections and
ByteTrack re-IDs them anyway.
"""

import numpy as np
import supervision as sv

import settings as cfg


class PlateTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=cfg.BYTETRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer=cfg.BYTETRACK_LOST_BUFFER,
            minimum_matching_threshold=cfg.BYTETRACK_MATCH_THRESHOLD,
            frame_rate=cfg.BYTETRACK_FRAME_RATE,
        )

    def update(self, boxes, scores):
        """
        boxes  : (N,4) xyxy float
        scores : (N,)  float
        Returns list of (xyxy_int_tuple, track_id).
        """
        if len(boxes) == 0:
            dets = sv.Detections.empty()
        else:
            dets = sv.Detections(
                xyxy=boxes.astype(np.float32),
                confidence=scores.astype(np.float32),
                class_id=np.zeros(len(boxes), dtype=int),
            )
        tracked = self.tracker.update_with_detections(dets)
        out = []
        for xyxy, tid in zip(tracked.xyxy, tracked.tracker_id):
            if tid is None:
                continue
            out.append((tuple(map(int, xyxy)), int(tid)))
        return out
