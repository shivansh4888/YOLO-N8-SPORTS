"""
src/annotator.py — Pro HUD Edition
────────────────────────────────────
Label resolution order (first match wins):
  1. Jersey # → PLAYER_ROSTER lookup         →  "#8  Ravindra Jadeja"
  2. Name text read from jersey back          →  "#8  JADEJA"
  3. Face cluster + jersey (if face_id on)    →  "#8 Player-A"
  4. Jersey number only                       →  "#8"
  5. Track ID fallback                        →  "ID 45"
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
        return (40, 190, 40)
    if kmh < 25:
        return (20, 190, 215)
    return (40, 50, 215)


def _resolve_label(tid: int, tracker, face_id, jersey_ocr=None) -> str:
    jersey_raw = tracker.id_jersey.get(int(tid), "")
    jersey_key = str(jersey_raw).strip()

    # 1. Roster lookup via jersey number
    if jersey_key:
        roster_name = config.PLAYER_ROSTER.get(jersey_key, "")
        if roster_name:
            return f"#{jersey_key}  {roster_name}"

    # 2. Name text read from jersey back (e.g. "JADEJA")
    if jersey_ocr is not None:
        back_name = jersey_ocr.get_name_from_back(int(tid))
        if back_name:
            return f"#{jersey_key}  {back_name}" if jersey_key else back_name

    # 3. face_id cluster label
    if face_id is not None:
        label = face_id.get_label(int(tid), jersey_key)
        if label:
            return label

    # 4. Jersey number only
    if jersey_key:
        return f"#{jersey_key}"

    # 5. Nothing known yet
    return ""


class Annotator:
    def __init__(self):
        self.font      = cv2.FONT_HERSHEY_SIMPLEX
        self.trail_len = config.TRAIL_LENGTH
        self.box_thick = config.BOX_THICKNESS
        self._jersey_ocr_ref = None

    def set_jersey_ocr(self, jersey_ocr):
        """Wire in the JerseyOCR instance so we can use back-name reads."""
        self._jersey_ocr_ref = jersey_ocr

    def annotate(self, frame, tracked, tracker, speed_est, face_id, frame_idx=0):
        out = frame.copy()

        if tracked.tracker_id is None or len(tracked) == 0:
            if config.SHOW_COUNT_OVERLAY:
                self._count(out, 0, frame_idx)
            return out

        for i, tid in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
            col = id_to_bgr(tid)

            if config.DRAW_TRAILS:
                trail = tracker.get_trail(tid, self.trail_len)
                segs  = self._trail_segs(trail)
                for j, seg in enumerate(segs):
                    a = (j + 1) / max(len(trail), 1)
                    cv2.line(out, seg[0], seg[1],
                             tuple(int(c * a) for c in col),
                             max(1, int(a * 3)), cv2.LINE_AA)

            cv2.rectangle(out, (x1, y1), (x2, y2), col, self.box_thick, cv2.LINE_AA)

            speed = speed_est.get_speed(tid) if speed_est else 0.0
            label = _resolve_label(int(tid), tracker, face_id, self._jersey_ocr_ref)
            self._hud(out, tid, label, speed, x1, y1, col)

        if config.SHOW_COUNT_OVERLAY:
            self._count(out, len(tracked), frame_idx)
        return out

    def _trail_segs(self, trail):
        return [
            ((int(trail[j - 1][0]), int(trail[j - 1][1])),
             (int(trail[j][0]),     int(trail[j][1])))
            for j in range(1, len(trail))
        ]

    def _hud(self, frame, tid, label, speed, x1, y1, col):
        F, s1, s2, th, pad = self.font, 0.50, 0.42, 1, 4
        row1 = label if label else f"ID {tid}"
        row2 = f"{speed:.1f} km/h" if speed > 0.5 else f"ID {tid}"

        (w1, h1), _ = cv2.getTextSize(row1, F, s1, th)
        (w2, h2), _ = cv2.getTextSize(row2, F, s2, th)
        bw  = max(w1, w2) + 2 * pad
        bh  = h1 + h2 + 3 * pad
        by2 = max(y1 - 2, bh)
        by1 = by2 - bh
        r1y2 = by1 + h1 + 2 * pad

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