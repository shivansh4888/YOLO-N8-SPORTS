# Sports Multi-Object Tracking (MOT) Pipeline

> Detect and persistently track players in public sports footage using YOLOv8 + ByteTrack.

---

## Demo Output

| Output | Description |
|---|---|
| `outputs/annotated_video.mp4` | Input video with bounding boxes, unique IDs, and motion trails |
| `outputs/heatmap.png` | Movement density heatmap over the full video |
| `outputs/trajectories.png` | All player paths plotted on a dark canvas |
| `outputs/count_over_time.png` | Active player count vs. time chart |

**Source video:** https://www.youtube.com/watch?v=0pYlJ7hA5Zo (ICC Cricket Highlights — public)

---

## Project Structure

```
sports-mot-tracker/
├── main.py                  # Entry point — orchestrates the full pipeline
├── config.py                # All hyperparameters in one place
├── download_video.py        # yt-dlp wrapper to fetch public video
├── requirements.txt         # All pip dependencies
├── report.md                # Technical write-up
│
├── src/
│   ├── detector.py          # YOLOv8 inference wrapper
│   ├── tracker.py           # ByteTrack ID assignment + history
│   ├── annotator.py         # Frame drawing (boxes, IDs, trails)
│   └── video_io.py          # OpenCV read/write with context managers
│
├── utils/
│   ├── heatmap.py           # Movement heatmap generator
│   └── stats.py             # Count plot + trajectory visualisation
│
├── data/                    # Input videos (git-ignored)
└── outputs/                 # All generated outputs (git-ignored)
```

---

## Installation

### 1. Clone / download the project
```bash
git clone https://github.com/your-username/sports-mot-tracker.git
cd sports-mot-tracker
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU acceleration (optional but recommended):**
> Install the CUDA build of PyTorch for ~10× speed:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Usage

### Step 1 — Download the test video
```bash
python download_video.py
# or use your own video:
python download_video.py --url "https://www.youtube.com/watch?v=YOUR_ID"
```

### Step 2 — Run the pipeline
```bash
python main.py
```

### Step 3 — Custom options
```bash
# Use a different video
python main.py --video data/my_clip.mp4

# Use a lighter/faster model
python main.py --model yolov8n.pt

# Process every other frame (faster)
python main.py --skip 1

# Disable motion trails
python main.py --no-trails
```

---

## Configuration

All tuneable parameters live in `config.py`. Key settings:

| Parameter | Default | Effect |
|---|---|---|
| `YOLO_MODEL` | `yolov8m.pt` | Model size (n/s/m/l/x = speed vs accuracy) |
| `CONFIDENCE_THRESHOLD` | `0.4` | Min detection confidence |
| `TRACK_BUFFER` | `30` | Frames a lost track survives before ID retirement |
| `FRAME_SKIP` | `1` | Process every Nth frame |
| `TRAIL_LENGTH` | `40` | Length of motion trail in frames |
| `INFERENCE_SIZE` | `1280` | Input resolution for YOLO |

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| `ultralytics` | ≥8.0 | YOLOv8 detection |
| `supervision` | ≥0.18 | ByteTrack + annotation helpers |
| `opencv-python-headless` | ≥4.8 | Video I/O |
| `numpy` | ≥1.24 | Array operations |
| `matplotlib` | ≥3.7 | Plots and heatmaps |
| `scipy` | latest | Gaussian blur for heatmap |
| `yt-dlp` | ≥2024.1 | Video download |
| `tqdm` | ≥4.65 | Progress bars |
| `torch` | ≥2.0 | Deep learning backend for YOLO |

---

## Assumptions

- Input video contains humans as the primary subjects to track.
- Video is publicly accessible and used for educational/research purposes.
- CPU inference is assumed by default; GPU is auto-used if available via PyTorch.
- Output video preserves the original FPS and resolution.

---

## Limitations

- ID switches can occur during heavy occlusion (multiple players colliding).
- Tracking degrades significantly if YOLO misses a player for >30 consecutive frames.
- No Re-ID module — players who leave and re-enter the frame get new IDs.
- Heatmap is in pixel-space, not pitch-space (no homography to top-view).
- Not tested on aerial drone footage or fisheye cameras.

---

## Possible Improvements

- Add a Re-ID appearance model (OSNet, StrongSORT) for cross-camera or re-entry tracking.
- Implement homography to project detections onto a top-down pitch view.
- Team clustering via jersey colour (k-means in HSV space).
- Speed estimation using homography + timestamps.
- Real-time mode using webcam or RTSP stream input.
- Model comparison: YOLOv8 vs RT-DETR vs YOLOv9.
