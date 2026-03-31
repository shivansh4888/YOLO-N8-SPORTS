"""
config.py — CPU-Optimised Full Feature Edition
───────────────────────────────────────────────
Every tunable parameter in one place.

WHAT'S NEW vs previous versions:
  • BoT-SORT tracker (camera-motion compensated, stable IDs on CPU)
  • Face clustering (InsightFace embeddings + DBSCAN)
  • Jersey OCR throttled to every 20 frames
  • Speed via homography (pure numpy)
  • All heavy ops are rate-limited for CPU viability

INTERVIEW QUESTION: "Why not StrongSORT on CPU?"
ANSWER: StrongSORT runs OSNet Re-ID on EVERY detection EVERY frame.
  On CPU that's ~200ms per crop. At 20 players × 30fps = 600 crops/sec
  → completely unusable. BoT-SORT skips the Re-ID network and instead
  uses camera motion compensation + better Kalman initialization,
  getting ~80% of the ID-stability benefit at ~1% of the compute cost.
"""

import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

INPUT_VIDEO        = os.path.join(DATA_DIR, "input_video.mp4")
OUTPUT_VIDEO       = os.path.join(OUTPUT_DIR, "annotated_video.mp4")
OUTPUT_HEATMAP     = os.path.join(OUTPUT_DIR, "heatmap.png")
OUTPUT_COUNT_PLOT  = os.path.join(OUTPUT_DIR, "count_over_time.png")
OUTPUT_TRAJECTORY  = os.path.join(OUTPUT_DIR, "trajectories.png")
OUTPUT_SPEED_PLOT  = os.path.join(OUTPUT_DIR, "speed_chart.png")
OUTPUT_FACE_GRID   = os.path.join(OUTPUT_DIR, "face_clusters.png")
OUTPUT_STATS_CSV   = os.path.join(OUTPUT_DIR, "player_stats.csv")

# ── Detection ─────────────────────────────────────────────────────
YOLO_MODEL           = "yolov8m.pt"   # m = best CPU speed/accuracy tradeoff
TARGET_CLASSES       = [0]            # person only
CONFIDENCE_THRESHOLD = 0.40
MIN_BOX_AREA         = 600            # px² — filter distant/tiny detections
INFERENCE_SIZE       = 1280           # longer side in pixels

# ── Tracking — BoT-SORT ───────────────────────────────────────────
# BoT-SORT: ByteTrack + camera-motion compensation (ECC homography)
# Built into Ultralytics — no extra install needed.
# Key advantage over ByteTrack: compensates for camera pan/zoom so
# predicted positions are more accurate → fewer ID switches.
TRACK_BUFFER         = 30    # frames a lost track survives
TRACK_IOU_THRESHOLD  = 0.30
BOTSORT_GMC          = True  # Global Motion Compensation (ECC algorithm)

# ── Speed estimation ─────────────────────────────────────────────
ENABLE_SPEED         = True
SPEED_SMOOTHING_WINDOW = 20   # rolling average window (frames)
MAX_SPEED_KMH        = 42.0   # clamp glitch spikes

# Calibration: 4 pixel↔real-world point pairs
# YOU MUST UPDATE PITCH_PIXEL_PTS for your video.
# Run: python utils/pick_points.py  to click them interactively.
# These defaults are approximate for a standard broadcast cricket angle.
PITCH_PIXEL_PTS = [
    (430,  90),   # top-left crease corner
    (710,  90),   # top-right crease corner
    (750, 510),   # bottom-right crease corner
    (390, 510),   # bottom-left crease corner
]
# Real-world metres — standard cricket pitch crease rectangle
PITCH_REAL_PTS = [
    (0.0,   0.0),
    (3.05,  0.0),
    (3.05, 20.12),
    (0.0,  20.12),
]

# ── Face detection + clustering ───────────────────────────────────
ENABLE_FACE_ID       = True

# InsightFace model — buffalo_sc is the lightweight CPU-friendly model.
# buffalo_l is more accurate but ~3× slower. Use buffalo_sc on CPU.
FACE_MODEL_PACK      = "buffalo_sc"   # auto-downloads ~10MB

# Face detection runs every N frames (heavy on CPU)
FACE_DETECT_INTERVAL = 10            # frames

# Minimum face size to attempt recognition (pixels, shorter side)
MIN_FACE_SIZE        = 40

# DBSCAN clustering parameters
# eps: cosine distance threshold for "same person" (0–2 scale)
# 0.4 = tight clusters (fewer merges), 0.6 = looser (fewer splits)
FACE_CLUSTER_EPS     = 0.45
FACE_CLUSTER_MIN_SAMPLES = 2         # min detections to form a cluster

# Save best face crop per cluster for the face grid output
SAVE_FACE_CROPS      = True

# ── Jersey OCR ───────────────────────────────────────────────────
ENABLE_JERSEY_OCR    = True
JERSEY_OCR_INTERVAL  = 20           # every 20 frames on CPU
JERSEY_CROP_PADDING  = 0.12
JERSEY_OCR_CONF      = 0.55

# Optional: map jersey numbers to names if you know them later
# Leave empty for "no roster" mode — shows #7, #18 etc.
PLAYER_ROSTER: dict = {}
# Example (add yours after seeing the jersey numbers detected):
# PLAYER_ROSTER = {"7": "MS Dhoni", "18": "Virat Kohli"}

# ── Video processing ─────────────────────────────────────────────
FRAME_SKIP    = 1      # process every other frame on CPU (saves ~40% time)
OUTPUT_CODEC  = "mp4v"

# ── Annotation ───────────────────────────────────────────────────
TRAIL_LENGTH       = 35
BOX_THICKNESS      = 2
DRAW_TRAILS        = True
SHOW_COUNT_OVERLAY = True
SHOW_SPEED         = True
SHOW_FACE_LABEL    = True
SHOW_JERSEY        = True
