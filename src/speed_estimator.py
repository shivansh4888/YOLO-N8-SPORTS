"""
src/speed_estimator.py
───────────────────────
Homography-based per-player speed estimation (km/h).
Pure numpy — zero GPU cost, runs in <1ms per frame.

CALIBRATION:  python utils/pick_points.py  → click 4 known pitch corners
ACCURACY:     ±2–4 km/h typical on broadcast footage
"""

import numpy as np
import cv2
from collections import defaultdict, deque
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class SpeedEstimator:

    def __init__(self, fps: float = 30.0):
        self.fps = max(fps, 1.0)
        self.H: np.ndarray = None
        self._calibrated = False
        self._world_pos:  dict[int, deque] = defaultdict(lambda: deque(maxlen=config.SPEED_SMOOTHING_WINDOW + 2))
        self._speed:      dict[int, float] = defaultdict(float)
        self._speed_buf:  dict[int, deque] = defaultdict(lambda: deque(maxlen=config.SPEED_SMOOTHING_WINDOW))
        self._peak_speed: dict[int, float] = defaultdict(float)

    def calibrate(self, pixel_pts=config.PITCH_PIXEL_PTS, real_pts=config.PITCH_REAL_PTS) -> bool:
        if len(pixel_pts) < 4:
            print("[Speed] Need >=4 calibration points. Speed disabled.")
            return False
        H, mask = cv2.findHomography(np.float32(pixel_pts), np.float32(real_pts), cv2.RANSAC, 5.0)
        if H is None:
            print("[Speed] Homography failed. Speed disabled.")
            return False
        self.H = H
        self._calibrated = True
        print(f"[Speed] Calibrated. {int(mask.sum()) if mask is not None else len(pixel_pts)} inliers.")
        return True

    def project(self, px: float, py: float) -> tuple:
        if not self._calibrated:
            return px, py
        world = cv2.perspectiveTransform(np.float32([[[px, py]]]), self.H)
        return float(world[0,0,0]), float(world[0,0,1])

    def update(self, track_id: int, cx: float, cy: float) -> float:
        if not self._calibrated:
            return 0.0
        tid = int(track_id)
        X, Y = self.project(cx, cy)
        hist = self._world_pos[tid]
        hist.append((X, Y))
        if len(hist) < 2:
            return 0.0
        dist_m = float(np.sqrt((X - hist[-2][0])**2 + (Y - hist[-2][1])**2))
        speed  = min((dist_m * self.fps) * 3.6, config.MAX_SPEED_KMH)
        self._speed_buf[tid].append(speed)
        smoothed = float(np.mean(self._speed_buf[tid]))
        self._speed[tid] = smoothed
        if smoothed > self._peak_speed[tid]:
            self._peak_speed[tid] = smoothed
        return smoothed

    def get_speed(self, track_id: int) -> float:
        return self._speed.get(int(track_id), 0.0)

    def get_max_speed(self, track_id: int) -> float:
        return self._peak_speed.get(int(track_id), 0.0)
