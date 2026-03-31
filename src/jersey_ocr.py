"""
src/jersey_ocr.py
──────────────────
Reads jersey numbers from player torso crops using EasyOCR.
Throttled to every JERSEY_OCR_INTERVAL frames for CPU viability.
Uses multi-frame voting for stable readings.
"""

import cv2
import numpy as np
from collections import defaultdict
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

try:
    import easyocr
    _EASYOCR_OK = True
except ImportError:
    _EASYOCR_OK = False
    print("[JerseyOCR] easyocr not found. Run: pip install easyocr")


class JerseyOCR:
    def __init__(self):
        self._reader = None
        self._votes: dict = defaultdict(lambda: defaultdict(float))
        self._available = _EASYOCR_OK and config.ENABLE_JERSEY_OCR
        if self._available:
            self._load()

    def _load(self):
        print("[JerseyOCR] Loading (first run downloads ~100MB)...")
        try:
            import torch
            self._reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
            print("[JerseyOCR] Ready.")
        except Exception as e:
            print(f"[JerseyOCR] Failed: {e}")
            self._available = False

    def read(self, frame: np.ndarray, bbox: np.ndarray, track_id: int) -> tuple:
        """Returns (jersey_number_str, confidence) or ('', 0.0)."""
        if not self._available:
            return "", 0.0
        crop = self._crop(frame, bbox)
        if crop is None:
            return "", 0.0
        try:
            res = self._reader.readtext(crop, allowlist='0123456789',
                                        detail=1, paragraph=False, min_size=8)
        except Exception:
            return "", 0.0
        valid = [(t.strip(), c) for (_, t, c) in res
                 if c >= config.JERSEY_OCR_CONF and t.strip().isdigit()
                 and 1 <= len(t.strip()) <= 3]
        if not valid:
            return "", 0.0
        best_t, best_c = max(valid, key=lambda x: x[1])
        self._votes[int(track_id)][best_t] += best_c
        return best_t, best_c

    def best(self, track_id: int) -> str:
        v = self._votes.get(int(track_id), {})
        return max(v, key=v.get) if v else ""

    def _crop(self, frame, bbox):
        h_f, w_f = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        px = int((x2 - x1) * config.JERSEY_CROP_PADDING)
        x1, x2 = max(0, x1 - px), min(w_f, x2 + px)
        bh = y2 - y1
        if bh < 25 or (x2 - x1) < 12:
            return None
        c = frame[y1 + int(bh * .2):y1 + int(bh * .7), x1:x2]
        if c.size == 0:
            return None
        up = cv2.resize(c, (c.shape[1] * 4, c.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
        g  = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        sh = cv2.filter2D(g, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
        bi = cv2.adaptiveThreshold(sh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(bi, cv2.COLOR_GRAY2BGR)

    @property
    def is_available(self): return self._available
