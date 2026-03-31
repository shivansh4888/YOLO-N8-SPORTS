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
ENABLE_FACE_ID       = False   # set True only if you add data/faces/ reference photos
FACE_MODEL_PACK      = "buffalo_sc"   # auto-downloads ~10MB
FACE_DETECT_INTERVAL = 10            # frames
MIN_FACE_SIZE        = 40
FACE_CLUSTER_EPS     = 0.45
FACE_CLUSTER_MIN_SAMPLES = 2
SAVE_FACE_CROPS      = True

# ── Jersey OCR ───────────────────────────────────────────────────
ENABLE_JERSEY_OCR    = True
JERSEY_OCR_INTERVAL  = 20           # every 20 frames on CPU
JERSEY_CROP_PADDING  = 0.12
JERSEY_OCR_CONF      = 0.55

# Optional: map jersey numbers to names.
# Leave empty — it is auto-loaded from data/roster.csv at startup.
# You can still hardcode here if you prefer:
# PLAYER_ROSTER = {"7": "MS Dhoni", "18": "Virat Kohli"}
PLAYER_ROSTER: dict = {}

# ── Face reference matching (Mode B) ─────────────────────────────
# Cosine similarity threshold to accept a reference photo match (0–1).
# 0.45 works well for broadcast footage.
# Raise to 0.50 if getting wrong names, lower to 0.40 if missing matches.
FACE_MATCH_THRESHOLD  = 0.45

# Minimum cumulative vote score before a name is confirmed.
# Each matching frame adds ~0.5–0.9 to the score.
# 3.0 ≈ ~5 good matching frames before the name is locked in.
FACE_MATCH_MIN_VOTES  = 3.0

# ── Video processing ─────────────────────────────────────────────
FRAME_SKIP    = 1      # process every Nth frame (1 = every frame, 2 = every other)
OUTPUT_CODEC  = "mp4v"

# ── Heatmap ──────────────────────────────────────────────────────
HEATMAP_SIGMA    = 8          # Gaussian blur sigma (higher = smoother)
HEATMAP_COLORMAP = "hot"      # "hot" | "jet" | "plasma"

# ── Annotation ───────────────────────────────────────────────────
TRAIL_LENGTH       = 35
BOX_THICKNESS      = 2
DRAW_TRAILS        = True
SHOW_COUNT_OVERLAY = True
SHOW_SPEED         = True
SHOW_FACE_LABEL    = True
SHOW_JERSEY        = True