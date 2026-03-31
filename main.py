"""
main.py — CPU Full-Feature MOT Pipeline
─────────────────────────────────────────
Features:
  [1] YOLOv8m detection
  [2] BoT-SORT tracking  (camera-motion compensated → stable IDs)
  [3] Speed estimation   (homography → km/h per player)
  [4] Jersey OCR         (EasyOCR every 20 frames → #number)
  [5] Face clustering    (InsightFace + DBSCAN → Player-A/B/C)
  [6] Pro HUD annotation (badge: name + jersey + speed)
  [7] Heatmap, trajectories, speed chart, player CSV

USAGE:
  python main.py
  python main.py --video data/myclip.mp4
  python main.py --no-face     (skip face ID, faster)
  python main.py --no-speed    (skip homography)
  python main.py --no-ocr      (skip jersey OCR)

INTERVIEW ANSWER — "walk me through your pipeline":
  Every frame goes through 4 stages:
  detect → track → enrich (speed + jersey + face) → annotate
  Each stage is a separate module → single responsibility principle.
  Heavy ops (OCR, face) are rate-limited for CPU viability.
  Post-processing generates analytics after the video loop.
"""

import argparse
import sys
import os
from src.botsort_tracker import BotSortTracker 
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.video_io        import VideoReader, VideoWriter
from src.annotator       import Annotator
from src.speed_estimator import SpeedEstimator
from src.jersey_ocr      import JerseyOCR
from src.face_identifier import FaceIdentifier
from src.botsort_tracker import BotSortTracker
from utils.heatmap       import generate_heatmap
from utils.stats         import (plot_count_over_time, plot_trajectories,
                                  print_summary)

# We import YOLO here to use its built-in track() with BoT-SORT
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Sports MOT — CPU Full Feature Pipeline")
    p.add_argument("--video",     default=config.INPUT_VIDEO)
    p.add_argument("--output",    default=config.OUTPUT_VIDEO)
    p.add_argument("--model",     default=config.YOLO_MODEL)
    p.add_argument("--skip",      type=int, default=config.FRAME_SKIP)
    p.add_argument("--no-speed",  action="store_true")
    p.add_argument("--no-ocr",    action="store_true")
    p.add_argument("--no-face",   action="store_true")
    p.add_argument("--no-trails", action="store_true")
    return p.parse_args()


def run(args):
    print("\n" + "═"*60)
    print("  SPORTS MOT — CPU FULL FEATURE PIPELINE")
    print(f"  Video  : {args.video}")
    print(f"  Output : {args.output}")
    print("═"*60 + "\n")

    # ── Initialise all modules ────────────────────────────────────
    print("[main] Loading YOLOv8 model...")
    model = YOLO(args.model)

    tracker   = BotSortTracker()
    annotator = Annotator()

    # Speed estimator
    speed_est = None
    if not args.no_speed and config.ENABLE_SPEED:
        speed_est = SpeedEstimator()
        if not speed_est.calibrate():
            print("[main] Speed estimation unavailable (update PITCH_PIXEL_PTS in config.py)")
            speed_est = None

    # Jersey OCR
    jersey_ocr = JerseyOCR() if not args.no_ocr else None
    if jersey_ocr and not jersey_ocr.is_available:
        jersey_ocr = None

    # Face identifier
    face_id = FaceIdentifier() if not args.no_face else None
    if face_id and not face_id.is_available:
        face_id = None

    print("\n[main] Module status:")
    print(f"  BoT-SORT tracker : ready")
    print(f"  Speed estimator  : {'ready' if speed_est else 'disabled'}")
    print(f"  Jersey OCR       : {'ready' if jersey_ocr else 'disabled'}")
    print(f"  Face ID          : {'ready' if face_id else 'disabled'}")
    print()

    reference_frame = None
    # Speed history for chart: track_id → [(frame_idx, speed)]
    speed_history: dict = {}

    # ── Main video loop ───────────────────────────────────────────
    with VideoReader(args.video) as reader:
        if speed_est:
            speed_est.fps = reader.fps

        meta  = reader.metadata
        total = reader.frame_count // max(args.skip + 1, 1)

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        with VideoWriter(args.output, reader.fps, reader.width, reader.height) as writer:

            for frame, frame_idx in tqdm(
                reader.frames(skip=args.skip),
                total=total, desc="Processing", unit="frame", ncols=80,
            ):
                if reference_frame is None:
                    reference_frame = frame.copy()

                # ── DETECT + TRACK (BoT-SORT via YOLO .track()) ──
                # persist=True keeps the tracker state between calls
                # tracker="botsort" uses BoT-SORT with GMC
                results = model.track(
                    frame,
                    persist=True,
                    tracker="botsort.yaml",
                    conf=config.CONFIDENCE_THRESHOLD,
                    classes=config.TARGET_CLASSES,
                    imgsz=config.INFERENCE_SIZE,
                    verbose=False,
                )
                tracked = tracker.update_from_results(results)

                if tracked.tracker_id is not None and len(tracked) > 0:

                    # ── SPEED ESTIMATION ─────────────────────────
                    if speed_est:
                        for i, tid in enumerate(tracked.tracker_id):
                            x1, y1, x2, y2 = tracked.xyxy[i]
                            cx, cy = (x1+x2)/2, (y1+y2)/2
                            spd = speed_est.update(int(tid), cx, cy)
                            if tid not in speed_history:
                                speed_history[tid] = []
                            speed_history[tid].append((frame_idx, spd))

                    # ── JERSEY OCR (every N frames) ───────────────
                    if jersey_ocr and frame_idx % config.JERSEY_OCR_INTERVAL == 0:
                        for i, tid in enumerate(tracked.tracker_id):
                            num, conf = jersey_ocr.read(frame, tracked.xyxy[i], tid)
                            if num:
                                tracker.assign_jersey(tid, num, conf)

                    # ── FACE ID (every N frames) ──────────────────
                    if face_id and frame_idx % config.FACE_DETECT_INTERVAL == 0:
                        face_id.process_frame(
                            frame,
                            tracked.xyxy,
                            tracked.tracker_id,
                        )

                    # ── Periodic face clustering ───────────────────
                    # Cluster every 100 frames so labels stabilise gradually
                    if face_id and frame_idx > 0 and frame_idx % 100 == 0:
                        face_id.cluster_and_label()

                # ── ANNOTATE ─────────────────────────────────────
                annotated = annotator.annotate(
                    frame, tracked, tracker, speed_est, face_id, frame_idx
                )

                # ── WRITE ─────────────────────────────────────────
                writer.write(annotated)

    # ── Final clustering pass ─────────────────────────────────────
    if face_id:
        print("\n[main] Running final face clustering...")
        face_id.cluster_and_label()
        face_id.save_face_grid(config.OUTPUT_FACE_GRID)

    # ── Post-processing analytics ─────────────────────────────────
    print("[main] Generating analytics outputs...")
    shape = (reader.height, reader.width)

    generate_heatmap(
        tracker.get_all_positions(), shape,
        reference_frame, config.OUTPUT_HEATMAP
    )
    plot_count_over_time(tracker.frame_counts, reader.fps, config.OUTPUT_COUNT_PLOT)
    plot_trajectories(tracker.track_history, shape, config.OUTPUT_TRAJECTORY)

    if speed_history:
        _save_speed_chart(speed_history, reader.fps)

    _export_csv(tracker, speed_est, face_id)
    print_summary(tracker.summary(), meta)

    print("\n[main] Done! Outputs:")
    outputs = [
        ("Annotated video ", args.output),
        ("Heatmap         ", config.OUTPUT_HEATMAP),
        ("Trajectories    ", config.OUTPUT_TRAJECTORY),
        ("Count plot      ", config.OUTPUT_COUNT_PLOT),
        ("Speed chart     ", config.OUTPUT_SPEED_PLOT),
        ("Face grid       ", config.OUTPUT_FACE_GRID),
        ("Player CSV      ", config.OUTPUT_STATS_CSV),
    ]
    for label, path in outputs:
        exists = "[ok]" if os.path.exists(path) else "[--]"
        print(f"  {exists} {label} → {path}")


def _save_speed_chart(speed_history: dict, fps: float):
    """Simple speed-over-time chart."""
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = plt.cm.tab20.colors
        for idx, (tid, readings) in enumerate(speed_history.items()):
            if not readings:
                continue
            times  = [r[0] / max(fps, 1) for r in readings]
            speeds = [r[1] for r in readings]
            ax.plot(times, speeds, color=colors[idx % 20], linewidth=0.9,
                    alpha=0.8, label=f"ID {tid}")
        ax.axhline(25, color="orange", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (km/h)")
        ax.set_ylim(0, config.MAX_SPEED_KMH + 5)
        ax.set_title("Player speeds over time", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if len(speed_history) <= 12:
            ax.legend(fontsize=7, loc="upper right")
        plt.tight_layout()
        os.makedirs(os.path.dirname(config.OUTPUT_SPEED_PLOT), exist_ok=True)
        plt.savefig(config.OUTPUT_SPEED_PLOT, dpi=130)
        plt.close()
        print(f"[main] Speed chart saved → {config.OUTPUT_SPEED_PLOT}")
    except Exception as e:
        print(f"[main] Speed chart failed: {e}")


def _export_csv(tracker, speed_est, face_id):
    """Export per-player stats to CSV."""
    try:
        import pandas as pd
        rows = []
        for tid, positions in tracker.track_history.items():
            jersey    = tracker.id_jersey.get(tid, "")
            name      = tracker.id_name.get(tid, "")
            face_lbl  = face_id.track_labels.get(tid, "") if face_id else ""
            label     = face_id.get_label(tid, jersey) if face_id else (f"#{jersey}" if jersey else "")
            max_spd   = speed_est.get_max_speed(tid) if speed_est else 0.0
            avg_spd   = speed_est.get_speed(tid)     if speed_est else 0.0
            rows.append({
                "track_id":       tid,
                "display_label":  label or f"ID {tid}",
                "jersey_number":  jersey,
                "player_name":    name,
                "face_cluster":   face_lbl,
                "frames_tracked": len(positions),
                "max_speed_kmh":  round(max_spd, 1),
                "avg_speed_kmh":  round(avg_spd, 1),
            })
        if not rows:
            return
        df = pd.DataFrame(rows).sort_values("frames_tracked", ascending=False)
        os.makedirs(os.path.dirname(config.OUTPUT_STATS_CSV), exist_ok=True)
        df.to_csv(config.OUTPUT_STATS_CSV, index=False)
        print(f"[main] Player CSV saved → {config.OUTPUT_STATS_CSV} ({len(df)} players)")
    except Exception as e:
        print(f"[main] CSV export failed: {e}")


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.video):
        print(f"\n[ERROR] Video not found: {args.video}")
        print("  Step 1: python download_video.py")
        print("  Step 2: python utils/pick_points.py  (click 4 pitch corners)")
        print("  Step 3: python main.py\n")
        sys.exit(1)
    run(args)
