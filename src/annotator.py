"""
src/annotator.py — Pro HUD Edition
────────────────────────────────────
Draws per-player HUD badge:

  ┌──────────────────┐
  │  MS Dhoni  #7    │  ← roster name (from jersey OCR) + jersey number
  │  24.3 km/h       │  ← speed (green/amber/red by intensity)
  └──────────────────┘
  [    bounding box  ]
  ~~~ fading trail ~~~

Label resolution order (first match wins):
  1. Roster name from PLAYER_ROSTER  →  "#7 MS Dhoni"
  2. Face cluster + jersey           →  "#7 Player-A"
  3. Face cluster alone              →  "Player-A"
  4. Jersey number alone             →  "#7"
  5. Track ID fallback               →  "ID 3"

Colour assignment: golden-angle HSV → unique colour per track ID.
Speed colour: green ≤10, amber 10–25, red >25 km/h.

FIX (2025):
  Previously the roster lookup only happened inside face_id.get_label(),
  which meant it was skipped entirely when face_id was None or disabled.
  The _resolve_label() helper now performs the full lookup chain
  independently of whether face identification is running.
"""

import cv2
import numpy as np
import supervision as sv
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def id_to_bgr(track_id: int) -> tuple:
    hue = int((int(track_id) * 137.508) % 360)
    hsv = np.uint8([[[hue // 2, 210, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def speed_bgr(kmh: float) -> tuple:
    if kmh < 10:
        return (40, 190, 40)    # green  — walking
    if kmh < 25:
        return (20, 190, 215)   # amber  — jogging
    return (40, 50, 215)        # red    — sprinting


def _resolve_label(tid: int, tracker, face_id) -> str:
    """
    Resolve the best display label for a track ID.

    Resolution order:
      1. Roster name (jersey OCR → PLAYER_ROSTER lookup)  →  "#7 MS Dhoni"
      2. face_id.get_label() if face_id active             →  "#7 Player-A"
      3. Jersey number alone                               →  "#7"
      4. Empty string  (caller falls back to "ID {tid}")

    The key fix: roster lookup is done HERE unconditionally, so names
    always appear even when face_id is disabled (ENABLE_FACE_ID=False).
    """
    jersey_raw = tracker.id_jersey.get(int(tid), "")
    jersey_key = str(jersey_raw).strip()

    # ── Priority 1: Roster name lookup ───────────────────────────
    # Works as long as jersey OCR has fired at least once for this track.
    if jersey_key:
        roster_name = config.PLAYER_ROSTER.get(jersey_key, "")
        if roster_name:
            return f"#{jersey_key}  {roster_name}"

    # ── Priority 2: face_id label (face cluster ± jersey) ────────
    if face_id is not None:
        label = face_id.get_label(int(tid), jersey_key)
        if label:
            return label

    # ── Priority 3: jersey number only ───────────────────────────
    if jersey_key:
        return f"#{jersey_key}"

    # ── Priority 4: nothing known yet ────────────────────────────
    return ""


class Annotator:
    def __init__(self):
        self.font      = cv2.FONT_HERSHEY_SIMPLEX
        self.trail_len = config.TRAIL_LENGTH
        self.box_thick = config.BOX_THICKNESS

    def annotate(self, frame, tracked, tracker, speed_est, face_id, frame_idx=0):
        out = frame.copy()

        if tracked.tracker_id is None or len(tracked) == 0:
            if config.SHOW_COUNT_OVERLAY:
                self._count(out, 0, frame_idx)
            return out

        for i, tid in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
            col = id_to_bgr(tid)

            # ── Trail ─────────────────────────────────────────────
            if config.DRAW_TRAILS:
                trail = tracker.get_trail(tid, self.trail_len)
                segs  = self._trail_segs(trail)
                for j, seg in enumerate(segs):
                    a = (j + 1) / max(len(trail), 1)
                    cv2.line(out, seg[0], seg[1],
                             tuple(int(c * a) for c in col),
                             max(1, int(a * 3)), cv2.LINE_AA)

            # ── Bounding box ──────────────────────────────────────
            cv2.rectangle(out, (x1, y1), (x2, y2), col, self.box_thick, cv2.LINE_AA)

            # ── Resolve label & speed, then draw HUD ──────────────
            speed = speed_est.get_speed(tid) if speed_est else 0.0
            label = _resolve_label(int(tid), tracker, face_id)
            self._hud(out, tid, label, speed, x1, y1, col)

        if config.SHOW_COUNT_OVERLAY:
            self._count(out, len(tracked), frame_idx)
        return out

    # ── Internal helpers ──────────────────────────────────────────

    def _trail_segs(self, trail):
        return [
            ((int(trail[j - 1][0]), int(trail[j - 1][1])),
             (int(trail[j][0]),     int(trail[j][1])))
            for j in range(1, len(trail))
        ]

    def _hud(self, frame, tid, label, speed, x1, y1, col):
        F, s1, s2, th, pad = self.font, 0.50, 0.42, 1, 4

        # Row 1: name / jersey / id
        row1 = label if label else f"ID {tid}"

        # Row 2: speed (green/amber/red), or plain ID if speed unavailable
        row2 = f"{speed:.1f} km/h" if speed > 0.5 else f"ID {tid}"

        (w1, h1), _ = cv2.getTextSize(row1, F, s1, th)
        (w2, h2), _ = cv2.getTextSize(row2, F, s2, th)
        bw  = max(w1, w2) + 2 * pad
        bh  = h1 + h2 + 3 * pad
        by2 = max(y1 - 2, bh)
        by1 = by2 - bh
        r1y2 = by1 + h1 + 2 * pad

        # Coloured name row + dark speed row
        cv2.rectangle(frame, (x1, by1),  (x1 + bw, r1y2), col,          -1)
        cv2.rectangle(frame, (x1, r1y2), (x1 + bw, by2),  (25, 25, 25), -1)
        cv2.putText(frame, row1, (x1 + pad, r1y2 - pad - 1), F, s1,
                    (255, 255, 255),  th, cv2.LINE_AA)
        cv2.putText(frame, row2, (x1 + pad, by2 - pad),      F, s2,
                    speed_bgr(speed), th, cv2.LINE_AA)

    def _count(self, frame, count, frame_idx):
        txt = f"Players: {count}   Frame: {frame_idx}"
        (tw, th), _ = cv2.getTextSize(txt, self.font, 0.60, 2)
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, txt, (8, th + 8), self.font, 0.60,
                    (255, 255, 255), 2, cv2.LINE_AA)