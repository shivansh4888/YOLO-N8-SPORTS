"""
src/jersey_ocr.py
──────────────────
Reads jersey numbers AND player name text from player crops using EasyOCR.
Throttled to every JERSEY_OCR_INTERVAL frames for CPU viability.
Uses multi-frame voting for stable readings.

FIXES (2025):
  - Number crop zone changed from (20%–70%) to (55%–95%) of bbox height
    → catches large numbers printed at the bottom of the jersey back.
  - Added NAME crop zone (30%–60%) → reads "JADEJA", "KOHLI" etc.
    from the player's back, mapped to roster or stored directly.
  - Improved preprocessing: HSV-based colour isolation before threshold
    → handles orange/red numbers on blue/white/green backgrounds.
  - Fallback: if name text is read directly from back, it is stored
    in tracker.id_name even without a matching roster entry.
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


def _enhance_for_ocr(crop: np.ndarray) -> np.ndarray:
    """
    Enhanced preprocessing for jersey OCR.
    Handles coloured numbers (orange, white, red) on coloured backgrounds.

    Strategy:
      1. Try colour isolation (orange/white/red mask) — best for sports jerseys.
      2. Sharpen + adaptive threshold as fallback.
    """
    if crop is None or crop.size == 0:
        return crop

    # Upscale first — EasyOCR needs at least 32px tall characters
    scale = max(1, 120 // max(crop.shape[0], 1))
    up = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                    interpolation=cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(up, cv2.COLOR_BGR2HSV)

    # ── Colour masks for common jersey number colours ─────────────
    # Orange numbers (like India #8)
    orange_mask = cv2.inRange(hsv, (5, 120, 120), (25, 255, 255))
    # White numbers
    white_mask  = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))
    # Red numbers (some jerseys)
    red_mask1   = cv2.inRange(hsv, (0, 120, 120), (10, 255, 255))
    red_mask2   = cv2.inRange(hsv, (170, 120, 120), (180, 255, 255))
    red_mask    = cv2.bitwise_or(red_mask1, red_mask2)
    # Yellow numbers
    yellow_mask = cv2.inRange(hsv, (22, 120, 120), (38, 255, 255))

    combined_mask = cv2.bitwise_or(
        cv2.bitwise_or(orange_mask, white_mask),
        cv2.bitwise_or(red_mask, yellow_mask)
    )

    # If colour mask has enough pixels, use it
    if cv2.countNonZero(combined_mask) > 30:
        # Dilate to connect digits
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        result = cv2.bitwise_and(up, up, mask=combined_mask)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # Make background white, digits black for OCR
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # ── Fallback: sharpening + adaptive threshold ─────────────────
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, sharp_kernel)
    binary = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


class JerseyOCR:
    def __init__(self):
        self._reader = None
        self._votes: dict = defaultdict(lambda: defaultdict(float))
        self._name_votes: dict = defaultdict(lambda: defaultdict(float))
        self._available = _EASYOCR_OK and config.ENABLE_JERSEY_OCR
        if self._available:
            self._load()

    def _load(self):
        print("[JerseyOCR] Loading EasyOCR (first run downloads ~100MB)...")
        try:
            import torch
            gpu = torch.cuda.is_available()
            # Load TWO readers: digits only + English letters (for name on back)
            self._reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
            print(f"[JerseyOCR] Ready. GPU={'yes' if gpu else 'no (CPU)'}")
        except Exception as e:
            print(f"[JerseyOCR] Failed to load: {e}")
            self._available = False

    def read(self, frame: np.ndarray, bbox: np.ndarray, track_id: int) -> tuple:
        """
        Read jersey number AND player name from a player crop.

        Returns:
            (jersey_number_str, confidence)  e.g. ("8", 0.92)
            Side-effect: stores name vote in self._name_votes if name found.
        """
        if not self._available or self._reader is None:
            return "", 0.0

        tid = int(track_id)

        # ── 1. Read jersey NUMBER (bottom 55%–95% of bbox) ────────
        num_crop = self._crop_number(frame, bbox)
        number, num_conf = "", 0.0
        if num_crop is not None:
            enhanced = _enhance_for_ocr(num_crop)
            try:
                res = self._reader.readtext(
                    enhanced,
                    allowlist='0123456789',
                    detail=1,
                    paragraph=False,
                    min_size=8,
                )
                valid = [
                    (t.strip(), c) for (_, t, c) in res
                    if c >= config.JERSEY_OCR_CONF
                    and t.strip().isdigit()
                    and 1 <= len(t.strip()) <= 3
                ]
                if valid:
                    number, num_conf = max(valid, key=lambda x: x[1])
                    self._votes[tid][number] += num_conf
            except Exception as e:
                pass

        # ── 2. Read player NAME from back (top 25%–55% of bbox) ───
        # Reads text like "JADEJA", "KOHLI" etc. from jersey back.
        # Only fires if OCR hasn't confirmed a name yet for this track.
        if not self._get_confirmed_name(tid):
            name_crop = self._crop_name(frame, bbox)
            if name_crop is not None:
                try:
                    # No allowlist — we want letters here
                    res = self._reader.readtext(
                        name_crop,
                        detail=1,
                        paragraph=False,
                        min_size=6,
                    )
                    for (_, text, conf) in res:
                        t = text.strip().upper()
                        # Must be 3–12 letters, no digits/symbols
                        if (conf >= 0.50
                                and t.isalpha()
                                and 3 <= len(t) <= 12):
                            self._name_votes[tid][t] += conf
                            # Check if this name is in roster values
                            self._try_match_name_to_roster(tid, t, conf)
                except Exception:
                    pass

        return number, num_conf

    def _try_match_name_to_roster(self, tid: int, text: str, conf: float):
        """
        Try to match OCR-read name text against PLAYER_ROSTER values.
        e.g.  "JADEJA" matches "Ravindra Jadeja" → store jersey key.
        """
        text_lower = text.lower()
        for jersey_key, full_name in config.PLAYER_ROSTER.items():
            # Check if OCR text appears in the player's name (surname match)
            if text_lower in full_name.lower() or full_name.lower() in text_lower:
                # Boost the vote for this jersey number as well
                # so the annotator can resolve it
                self._votes[tid][jersey_key] += conf * 0.5
                break

    def _get_confirmed_name(self, tid: int) -> str:
        """Return best name text read from jersey back for this track, or ''."""
        v = self._name_votes.get(tid, {})
        if not v:
            return ""
        best = max(v, key=v.get)
        return best if v[best] >= 1.0 else ""  # need >1 vote to confirm

    def get_name_from_back(self, track_id: int) -> str:
        """Public: get player name OCR'd from jersey back. Empty if unsure."""
        return self._get_confirmed_name(int(track_id))

    def best(self, track_id: int) -> str:
        """Best jersey number for a track ID."""
        v = self._votes.get(int(track_id), {})
        return max(v, key=v.get) if v else ""

    # ── Crop helpers ──────────────────────────────────────────────

    def _crop_number(self, frame: np.ndarray, bbox: np.ndarray):
        """
        Crop the LOWER portion of the player bbox where the jersey number lives.
        Uses 55%–95% of the bbox height (changed from 20%–70%).
        """
        h_f, w_f = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        px = int((x2 - x1) * config.JERSEY_CROP_PADDING)
        x1c, x2c = max(0, x1 - px), min(w_f, x2 + px)
        bh = y2 - y1
        if bh < 30 or (x2c - x1c) < 15:
            return None
        # Bottom 55%–95% — where the number is on most cricket/football jerseys
        top    = y1 + int(bh * 0.55)
        bottom = y1 + int(bh * 0.95)
        bottom = min(bottom, h_f)
        crop = frame[top:bottom, x1c:x2c]
        return crop if crop.size > 0 else None

    def _crop_name(self, frame: np.ndarray, bbox: np.ndarray):
        """
        Crop the UPPER-MIDDLE portion of the player bbox where the name is.
        Uses 25%–55% of the bbox height.
        """
        h_f, w_f = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        px = int((x2 - x1) * config.JERSEY_CROP_PADDING)
        x1c, x2c = max(0, x1 - px), min(w_f, x2 + px)
        bh = y2 - y1
        if bh < 40 or (x2c - x1c) < 15:
            return None
        # Middle band — where surname is printed on jersey back
        top    = y1 + int(bh * 0.25)
        bottom = y1 + int(bh * 0.55)
        bottom = min(bottom, h_f)
        crop = frame[top:bottom, x1c:x2c]
        return crop if crop.size > 0 else None

    @property
    def is_available(self):
        return self._available