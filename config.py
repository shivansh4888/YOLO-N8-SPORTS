"""
config.py
─────────
Central configuration for the Multi-Object Tracking pipeline.

WHY A SEPARATE CONFIG FILE?
• Single place to tune everything — no hunting through 10 files.
• Makes ablation studies easy: change one number, re-run, compare.
• Interview answer: "Separation of concerns — logic vs parameters."

INTERVIEW QUESTIONS YOU SHOULD KNOW:
Q: Why 0.4 confidence threshold?
A: Lower → more detections but more false positives.
   Higher → fewer false positives but misses occluded players.
   0.4 is a good balance for sports footage with fast motion.

Q: What is IoU threshold in tracking?
A: Intersection over Union — measures bbox overlap between frames.
   Higher IoU threshold = stricter match = more ID switches.
   Lower = looser match = merges different people accidentally.

Q: Why process every 2nd frame (FRAME_SKIP=1)?
A: At 30fps, consecutive frames are nearly identical.
   Skipping reduces compute with minimal tracking loss.
   ByteTrack handles the gap using Kalman filter prediction.
"""

import os

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

# Input video — place your downloaded clip here
INPUT_VIDEO = os.path.join(DATA_DIR, "input_video.mp4")

# Output files
OUTPUT_VIDEO      = os.path.join(OUTPUT_DIR, "annotated_video.mp4")
OUTPUT_HEATMAP    = os.path.join(OUTPUT_DIR, "heatmap.png")
OUTPUT_COUNT_PLOT = os.path.join(OUTPUT_DIR, "count_over_time.png")
OUTPUT_TRAJECTORY = os.path.join(OUTPUT_DIR, "trajectories.png")

# ─────────────────────────────────────────────────────────────────
# DETECTION — YOLOv8
# ─────────────────────────────────────────────────────────────────

# Model weights — 'yolov8m.pt' balances speed vs accuracy.
# Options: yolov8n (nano/fastest), yolov8s, yolov8m, yolov8l, yolov8x (largest/best)
YOLO_MODEL = "yolov8m.pt"

# Only detect people (COCO class 0). Change to [] for all classes.
TARGET_CLASSES = [0]  # 0 = person in COCO dataset

# Minimum confidence to accept a detection (0.0 – 1.0)
CONFIDENCE_THRESHOLD = 0.4

# Minimum bounding box area in pixels — filters tiny false detections
MIN_BOX_AREA = 500  # pixels²

# ─────────────────────────────────────────────────────────────────
# TRACKING — ByteTrack
# ─────────────────────────────────────────────────────────────────

# How many frames a track can be "lost" before its ID is retired
TRACK_BUFFER = 30   # frames — at 30fps this = 1 second of memory

# IoU threshold for matching detections to existing tracks
TRACK_IOU_THRESHOLD = 0.3

# Minimum consecutive frames before a new track gets a stable ID
MIN_TRACK_LENGTH = 3

# ─────────────────────────────────────────────────────────────────
# VIDEO PROCESSING
# ─────────────────────────────────────────────────────────────────

# Process every Nth frame. 0 = every frame. 1 = every other frame.
# Higher = faster but potentially jerkier tracking.
FRAME_SKIP = 1

# Resize frame before inference (smaller = faster, less accurate)
# None = use original resolution
INFERENCE_SIZE = 1280  # pixels for the longer side

# Output video codec
OUTPUT_CODEC = "mp4v"

# ─────────────────────────────────────────────────────────────────
# ANNOTATION / VISUALISATION
# ─────────────────────────────────────────────────────────────────

# Trail length — how many past positions to draw as a motion trail
TRAIL_LENGTH = 40  # frames

# Bounding box line thickness
BOX_THICKNESS = 2  # pixels

# ID label font scale
LABEL_FONT_SCALE = 0.6

# Whether to draw trajectory trails on output video
DRAW_TRAILS = True

# Whether to show player count overlay on each frame
SHOW_COUNT_OVERLAY = True

# ─────────────────────────────────────────────────────────────────
# HEATMAP
# ─────────────────────────────────────────────────────────────────

# Gaussian blur sigma for heatmap smoothing
HEATMAP_SIGMA = 25

# Heatmap colormap — 'hot', 'jet', 'plasma' are popular
HEATMAP_COLORMAP = "hot"
