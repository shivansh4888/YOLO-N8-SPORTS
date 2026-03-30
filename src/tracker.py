"""
src/tracker.py
──────────────
Multi-Object Tracking using ByteTrack via the Supervision library.

THE CORE TRACKING PROBLEM — understand this deeply for interviews:

  Frame 1: Person A at (100,200), Person B at (400,200)
  Frame 2: Person A at (105,205), Person B at (395,205)
  → Easy: small IoU overlap → match A→A, B→B

  Frame 3: Person A walks behind Person B (OCCLUSION)
  Frame 4: Two people emerge — which is A? which is B?
  → Hard: need motion prediction + appearance cues

WHY BYTETRACK OVER SORT / DEEPSORT?

  SORT (2016):
    • Uses Kalman filter for motion prediction + Hungarian algorithm
      for detection-to-track assignment.
    • Simple and fast but loses IDs on occlusion often.

  DeepSORT (2017):
    • Adds a Re-ID appearance feature (deep CNN embedding).
    • More robust but needs a separate Re-ID model.
    • Slow — not suitable for real-time.

  ByteTrack (2022) — what we use:
    • Key insight: don't throw away low-confidence detections!
    • Round 1: match high-conf detections to tracks (IoU matching).
    • Round 2: match REMAINING tracks to LOW-conf detections.
    • The low-conf box that half-overlaps an occluded track is the
      exact signal needed to keep the ID alive through occlusion.
    • Result: near-SOTA accuracy with SORT-like speed. No Re-ID net needed.

KALMAN FILTER (what the tracker uses internally):
  A Kalman filter predicts where a tracked object WILL be next frame
  based on its velocity. When detection is missing, the filter
  "coasts" using prediction alone. This is why IDs survive short
  occlusions — the track exists in predicted space even with no detection.

HUNGARIAN ALGORITHM:
  Solves the assignment problem: given N detections and M tracks,
  find the minimum-cost matching. Cost = 1 - IoU (or distance).
  Runs in O(n³) — fast enough for ≤100 players per frame.
"""

import numpy as np
import supervision as sv
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class Tracker:
    """
    Wraps supervision's ByteTrack and maintains the full track history.

    TRACK HISTORY:
      For each ID we store a list of (cx, cy) centre-points over time.
      Used for:  trajectory drawing, heatmap generation, speed estimation.
    """

    def __init__(
        self,
        track_buffer: int = config.TRACK_BUFFER,
        iou_threshold: float = config.TRACK_IOU_THRESHOLD,
    ):
        """
        Args:
            track_buffer:  Frames a track survives without a detection
                           match before its ID is retired.
            iou_threshold: Minimum IoU overlap to accept a match between
                           a detection and an existing track.
        """
        # supervision's ByteTracker
        self.tracker = sv.ByteTrack(
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=iou_threshold,
        )

        # ── Track history store ───────────────────────────────────
        # dict mapping:  track_id (int) → list of (cx, cy) tuples
        # Grows over the whole video — used for heatmap + trajectories
        self.track_history: dict[int, list[tuple[float, float]]] = {}

        # For statistics: per-frame active count
        self.frame_counts: list[int] = []

        print(f"[Tracker] ByteTrack ready. buffer={track_buffer}, iou={iou_threshold}")

    def update(self, detections_raw: np.ndarray, frame: np.ndarray) -> sv.Detections:
        """
        Feed new frame detections into ByteTrack and get back tracked objects.

        Args:
            detections_raw: np.ndarray (N, 6) — [x1,y1,x2,y2,conf,cls_id]
                            as returned by Detector.detect().
            frame:          Original BGR frame (needed by supervision internals).

        Returns:
            tracked: sv.Detections object with .tracker_id assigned.
                     Access bboxes via tracked.xyxy,
                     IDs via tracked.tracker_id.

        DATA FLOW (important for interviews):
          raw_detections → supervision Detections → ByteTracker.update()
          → matched+predicted tracks → update our history → return
        """
        h, w = frame.shape[:2]

        # ── Convert raw numpy → supervision Detections ────────────
        if len(detections_raw) == 0:
            # No detections this frame — pass empty Detections
            sv_dets = sv.Detections.empty()
        else:
            sv_dets = sv.Detections(
                xyxy       = detections_raw[:, :4],          # [x1,y1,x2,y2]
                confidence = detections_raw[:, 4],           # conf scores
                class_id   = detections_raw[:, 5].astype(int),  # class ids
            )

        # ── Run ByteTrack ─────────────────────────────────────────
        # This is where the magic happens:
        #   1. Kalman filter predicts where each existing track should be.
        #   2. Hungarian algorithm matches predictions ↔ detections.
        #   3. Low-conf detections rescue tracks that had no high-conf match.
        #   4. Unmatched tracks → tentative → deleted after track_buffer frames.
        #   5. New detections not matching any track → new track + new ID.
        tracked = self.tracker.update_with_detections(sv_dets)

        # ── Update track history ──────────────────────────────────
        # For every tracked object, record its centre point this frame.
        if tracked.tracker_id is not None:
            for i, track_id in enumerate(tracked.tracker_id):
                x1, y1, x2, y2 = tracked.xyxy[i]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append((cx, cy))

        # ── Record active count ───────────────────────────────────
        active = len(tracked) if tracked.tracker_id is not None else 0
        self.frame_counts.append(active)

        return tracked

    def get_trail(self, track_id: int, length: int = config.TRAIL_LENGTH) -> list:
        """
        Returns the most recent `length` positions for a given track ID.
        Used by the annotator to draw motion trails.

        Returns list of (cx, cy) tuples, most recent last.
        """
        history = self.track_history.get(track_id, [])
        return history[-length:]  # last N points

    def get_all_positions(self) -> list[tuple[float, float]]:
        """
        Returns ALL positions for ALL tracks ever seen.
        Used to build the movement heatmap.
        """
        all_pts = []
        for positions in self.track_history.values():
            all_pts.extend(positions)
        return all_pts

    def summary(self) -> dict:
        """Returns a summary for logging and the technical report."""
        return {
            "total_unique_ids": len(self.track_history),
            "total_frames_processed": len(self.frame_counts),
            "avg_detections_per_frame": (
                round(sum(self.frame_counts) / len(self.frame_counts), 2)
                if self.frame_counts else 0
            ),
            "max_simultaneous": max(self.frame_counts) if self.frame_counts else 0,
        }
