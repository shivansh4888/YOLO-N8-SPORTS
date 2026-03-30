"""
main.py
───────
Pipeline orchestrator — ties all modules together.

This is the ENTRY POINT. Running this file:
  1. Loads config
  2. Opens the input video
  3. Initialises YOLOv8 detector + ByteTrack tracker + annotator
  4. Processes every frame: detect → track → annotate → write
  5. Generates post-processing outputs: heatmap, trajectories, count plot
  6. Prints a summary

USAGE:
    python main.py
    python main.py --video data/my_clip.mp4
    python main.py --video data/clip.mp4 --skip 2 --no-trails

DESIGN PATTERN — "orchestrator" main:
  main.py knows WHAT to do and in what ORDER.
  It does NOT know HOW each step works — that lives in the modules.
  This is the "single responsibility principle" applied to main files.

INTERVIEW ANSWER TO "WALK ME THROUGH YOUR CODE":
  Start here. Trace the data flow:
  frame (numpy) → Detector → raw detections (numpy)
  raw detections → Tracker → tracked Detections (supervision)
  tracked Detections → Annotator → annotated frame (numpy)
  annotated frame → VideoWriter → output.mp4
"""

import argparse
import sys
import os
from tqdm import tqdm  # progress bar — shows frame-processing progress

# ── Add project root to Python path ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.detector  import Detector
from src.tracker   import Tracker
from src.annotator import Annotator
from src.video_io  import VideoReader, VideoWriter
from utils.heatmap import generate_heatmap
from utils.stats   import plot_count_over_time, plot_trajectories, print_summary


def parse_args():
    """Command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Multi-Object Tracking for Sports Footage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video", type=str, default=config.INPUT_VIDEO,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output", type=str, default=config.OUTPUT_VIDEO,
        help="Path for annotated output video",
    )
    parser.add_argument(
        "--skip", type=int, default=config.FRAME_SKIP,
        help="Process every (skip+1)th frame. 0=every frame.",
    )
    parser.add_argument(
        "--no-trails", action="store_true",
        help="Disable motion trail drawing",
    )
    parser.add_argument(
        "--model", type=str, default=config.YOLO_MODEL,
        help="YOLOv8 model weights (yolov8n/s/m/l/x.pt)",
    )
    return parser.parse_args()


def run_pipeline(args) -> None:
    """
    Main pipeline execution.

    DATA FLOW PER FRAME:
      raw_frame
        ↓  Detector.detect()
      detections: np.ndarray (N, 6)  [x1,y1,x2,y2,conf,cls]
        ↓  Tracker.update()
      tracked: sv.Detections  (with .tracker_id assigned)
        ↓  Annotator.annotate()
      annotated_frame: np.ndarray
        ↓  VideoWriter.write()
      saved to output.mp4
    """

    print("\n" + "═" * 60)
    print("  MULTI-OBJECT TRACKING PIPELINE")
    print("  Model    :", args.model)
    print("  Input    :", args.video)
    print("  Output   :", args.output)
    print("  Frame skip:", args.skip)
    print("═" * 60 + "\n")

    # ── Initialise pipeline components ───────────────────────────
    detector  = Detector(model_path=args.model)
    tracker   = Tracker()
    annotator = Annotator(draw_trails=not args.no_trails)

    # Keep a reference frame for heatmap overlay (use first frame)
    reference_frame = None
    total_frames_written = 0

    # ── Open video reader ─────────────────────────────────────────
    with VideoReader(args.video) as reader:
        meta = reader.metadata

        # Open writer with same resolution and FPS as input
        # Note: if FRAME_SKIP > 0, effective FPS of output is lower.
        # We keep original FPS so playback speed is preserved (slow motion effect removed).
        with VideoWriter(args.output, reader.fps, reader.width, reader.height) as writer:

            # tqdm wraps the generator to show a live progress bar
            frame_gen = reader.frames(skip=args.skip)
            total_expected = reader.frame_count // (args.skip + 1)

            for frame, frame_idx in tqdm(
                frame_gen,
                total=total_expected,
                desc="Processing frames",
                unit="frame",
                ncols=80,
            ):
                # ── Save first frame for heatmap background ───────
                if reference_frame is None:
                    reference_frame = frame.copy()

                # ── STEP 1: Detect ────────────────────────────────
                detections = detector.detect(frame)

                # ── STEP 2: Track ─────────────────────────────────
                tracked = tracker.update(detections, frame)

                # ── STEP 3: Annotate ──────────────────────────────
                annotated = annotator.annotate(
                    frame, tracked, tracker, frame_idx
                )

                # ── STEP 4: Write ─────────────────────────────────
                writer.write(annotated)
                total_frames_written += 1

    # ── Post-processing: generate analytics outputs ───────────────
    print("\n[Pipeline] Generating post-processing outputs...")

    frame_shape = (reader.height, reader.width)

    # 1. Movement heatmap
    all_positions = tracker.get_all_positions()
    generate_heatmap(
        positions=all_positions,
        frame_shape=frame_shape,
        reference_frame=reference_frame,
        output_path=config.OUTPUT_HEATMAP,
    )

    # 2. Count over time plot
    plot_count_over_time(
        frame_counts=tracker.frame_counts,
        fps=reader.fps,
        output_path=config.OUTPUT_COUNT_PLOT,
    )

    # 3. Trajectory visualisation
    plot_trajectories(
        track_history=tracker.track_history,
        frame_shape=frame_shape,
        output_path=config.OUTPUT_TRAJECTORY,
    )

    # 4. Summary
    print_summary(tracker.summary(), meta)

    print(f"[Pipeline] All done!")
    print(f"  Annotated video  → {args.output}")
    print(f"  Heatmap          → {config.OUTPUT_HEATMAP}")
    print(f"  Count plot       → {config.OUTPUT_COUNT_PLOT}")
    print(f"  Trajectories     → {config.OUTPUT_TRAJECTORY}\n")


if __name__ == "__main__":
    args = parse_args()

    # Validate input file exists
    if not os.path.exists(args.video):
        print(f"[ERROR] Input video not found: {args.video}")
        print("  Run:  python download_video.py")
        print("  Or:   python main.py --video /path/to/your/video.mp4")
        sys.exit(1)

    run_pipeline(args)
