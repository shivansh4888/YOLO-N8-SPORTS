"""
src/annotator.py
────────────────
Draws all visual overlays onto video frames:
  • Colour-coded bounding boxes (unique colour per track ID)
  • Track ID label with background pill
  • Motion trail (last N centre-points as a fading polyline)
  • Active player count overlay (top-left corner)

WHY UNIQUE COLOURS PER ID?
  Human eye easily distinguishes colour — so if ID 3 is always cyan
  and ID 7 is always orange, you can follow individuals instantly
  without reading the number. Critical for sports analysis UX.

THE COLOUR ASSIGNMENT TRICK (interview-worthy):
  We use HSV colour space, not RGB. Hue = 0°–360°.
  Map track_id → hue by:  hue = (track_id * golden_ratio_angle) % 360
  The golden ratio angle (137.5°) distributes colours maximally
  far apart from each other, so consecutive IDs never look similar.
  Convert HSV → BGR for OpenCV rendering.
"""

import cv2
import numpy as np
import supervision as sv
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Golden-angle colour generator ─────────────────────────────────
# This function maps any integer ID to a visually distinct BGR colour.

def id_to_color(track_id: int) -> tuple[int, int, int]:
    """
    Maps a track ID to a unique, visually distinct BGR colour.

    Uses the golden angle (137.508°) in HSV hue space so that
    adjacent IDs get maximally separated hues.

    Returns: (B, G, R) tuple of uint8 values for OpenCV.
    """
    GOLDEN_ANGLE = 137.508  # degrees — derived from golden ratio
    hue = int((track_id * GOLDEN_ANGLE) % 360)

    # HSV → BGR via OpenCV
    # Create a 1×1 HSV pixel and convert it
    hsv_pixel = np.uint8([[[hue // 2, 220, 230]]])  # hue/2 because OpenCV uses 0-180
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


class Annotator:
    """
    Draws all visual overlays onto a copy of each video frame.

    Design choice: we always draw on a COPY of the frame, never
    mutating the original. This lets callers keep the clean frame
    if needed (e.g., for re-processing or saving separately).
    """

    def __init__(
        self,
        draw_trails: bool = config.DRAW_TRAILS,
        trail_length: int = config.TRAIL_LENGTH,
        box_thickness: int = config.BOX_THICKNESS,
        show_count: bool = config.SHOW_COUNT_OVERLAY,
    ):
        self.draw_trails  = draw_trails
        self.trail_length = trail_length
        self.box_thickness = box_thickness
        self.show_count   = show_count

    def annotate(
        self,
        frame: np.ndarray,
        tracked: sv.Detections,
        tracker,  # Tracker instance — for trail history
        frame_idx: int = 0,
    ) -> np.ndarray:
        """
        Draw all annotations on a copy of the frame.

        Args:
            frame:     BGR frame (H, W, 3).
            tracked:   sv.Detections with .xyxy and .tracker_id.
            tracker:   The Tracker instance (for trail history).
            frame_idx: Current frame index (used for count overlay).

        Returns:
            annotated: BGR frame with all overlays drawn.
        """
        annotated = frame.copy()

        if tracked.tracker_id is None or len(tracked) == 0:
            if self.show_count:
                self._draw_count_overlay(annotated, 0, frame_idx)
            return annotated

        # ── Draw each tracked person ──────────────────────────────
        for i, track_id in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
            color = id_to_color(track_id)

            # ── Motion trail ──────────────────────────────────────
            if self.draw_trails:
                trail = tracker.get_trail(track_id, self.trail_length)
                self._draw_trail(annotated, trail, color)

            # ── Bounding box ──────────────────────────────────────
            cv2.rectangle(
                annotated,
                (x1, y1), (x2, y2),
                color,
                self.box_thickness,
                lineType=cv2.LINE_AA,  # anti-aliased = smoother edges
            )

            # ── ID label with background ──────────────────────────
            self._draw_label(annotated, track_id, x1, y1, color)

        # ── Player count overlay ──────────────────────────────────
        if self.show_count:
            self._draw_count_overlay(annotated, len(tracked), frame_idx)

        return annotated

    # ── Private drawing helpers ────────────────────────────────────

    def _draw_trail(
        self,
        frame: np.ndarray,
        trail: list[tuple[float, float]],
        color: tuple[int, int, int],
    ) -> None:
        """
        Draws a fading polyline trail.

        WHY FADING?
        Opacity decreases for older positions — this gives the viewer
        a sense of direction AND speed (widely spaced points = fast).
        """
        if len(trail) < 2:
            return

        for j in range(1, len(trail)):
            pt1 = (int(trail[j - 1][0]), int(trail[j - 1][1]))
            pt2 = (int(trail[j][0]),     int(trail[j][1]))

            # Opacity: older points are more transparent
            alpha = j / len(trail)           # 0.0 (oldest) → 1.0 (newest)
            thickness = max(1, int(alpha * 3))  # thinner for older points

            # Blend colour with frame background for fading effect
            fade_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, pt1, pt2, fade_color, thickness, lineType=cv2.LINE_AA)

    def _draw_label(
        self,
        frame: np.ndarray,
        track_id: int,
        x1: int, y1: int,
        color: tuple[int, int, int],
    ) -> None:
        """
        Draws a filled pill background + white ID text above the bounding box.

        Design: coloured background matching the box → visually cohesive.
        """
        label   = f"ID {track_id}"
        font    = cv2.FONT_HERSHEY_SIMPLEX
        scale   = config.LABEL_FONT_SCALE
        thick   = 1

        # Measure text size to size the background rectangle
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
        pad = 3

        # Background rectangle
        bg_x1 = x1
        bg_y1 = max(y1 - th - 2 * pad - baseline, 0)
        bg_x2 = x1 + tw + 2 * pad
        bg_y2 = max(y1, th + baseline)

        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)  # -1 = filled

        # White text on coloured background
        cv2.putText(
            frame, label,
            (x1 + pad, max(y1 - pad - baseline, th)),
            font, scale, (255, 255, 255), thick, cv2.LINE_AA,
        )

    def _draw_count_overlay(
        self,
        frame: np.ndarray,
        count: int,
        frame_idx: int,
    ) -> None:
        """
        Draws a semi-transparent black bar at top-left showing player count.
        """
        text   = f"Players: {count}   Frame: {frame_idx}"
        font   = cv2.FONT_HERSHEY_SIMPLEX
        scale  = 0.7
        thick  = 2
        color  = (255, 255, 255)  # white text

        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        pad = 8

        # Semi-transparent overlay (black rectangle, alpha blended)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (tw + 2 * pad, th + 2 * pad), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)  # 50% transparent

        cv2.putText(frame, text, (pad, th + pad), font, scale, color, thick, cv2.LINE_AA)
