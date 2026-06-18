"""
rapid_ocr.py
============
Plate OCR using RapidOCR (PP-OCR recognition models run through onnxruntime).
No PaddlePaddle -> no aarch64 segfault on the Pi.

We feed an already-cropped plate, so detection/classification are off and only
the recognition head runs (use_rec=True). First run downloads ~15 MB of models.
"""

import re
from rapidocr import RapidOCR


class PlateOCR:
    def __init__(self):
        self.engine = RapidOCR()

    def read(self, crop):
        """Recognise text on a cropped plate. Returns an UPPER A-Z0-9 string ('' on failure)."""
        try:
            res = self.engine(crop, use_det=False, use_cls=False, use_rec=True)
            if res is None or not res.txts:
                return ""
            return re.sub(r"[^A-Z0-9]", "", str(res.txts[0]).upper())
        except Exception as e:
            print(f"[OCR error] {e}")
            return ""
