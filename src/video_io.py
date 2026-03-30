"""
src/video_io.py
───────────────
Handles all video file I/O using OpenCV.

RESPONSIBILITIES:
  • Open an input video and expose frame-by-frame iteration.
  • Create an output video writer with matching resolution/FPS.
  • Report video metadata (resolution, FPS, frame count).
  • Ensure files are properly released (no corrupted outputs).

WHY CONTEXT MANAGERS (with ... as ...)?
  VideoCapture and VideoWriter are OS-level resources.
  If your code crashes mid-processing, Python's garbage collector
  might not release the file handle in time → corrupted output.
  Using __enter__ / __exit__ guarantees cleanup even on exceptions.
  This is the professional pattern — always use it in interviews.

OPENCV COLOUR FORMAT:
  OpenCV reads frames as BGR (Blue-Green-Red), not RGB.
  Most other libraries (matplotlib, PIL, deep learning models)
  use RGB. Always convert with cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  when displaying with matplotlib or feeding to torchvision.
  YOLOv8's Ultralytics wrapper handles this internally.
"""

import cv2
import numpy as np
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class VideoReader:
    """
    Reads a video file frame by frame.

    Usage:
        with VideoReader("input.mp4") as reader:
            for frame in reader:
                process(frame)

    Or with frame skipping:
        for frame, idx in reader.frames(skip=1):
            ...
    """

    def __init__(self, path: str):
        self.path = str(path)
        self._cap = None
        self.fps         = 0.0
        self.width       = 0
        self.height      = 0
        self.frame_count = 0

    def __enter__(self):
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")

        self.fps         = self._cap.get(cv2.CAP_PROP_FPS)
        self.width       = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height      = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[VideoReader] Opened: {self.path}")
        print(f"  Resolution : {self.width}×{self.height}")
        print(f"  FPS        : {self.fps:.2f}")
        print(f"  Frames     : {self.frame_count}")
        return self

    def __exit__(self, *_):
        if self._cap:
            self._cap.release()
            print("[VideoReader] Released.")

    def __iter__(self):
        """Iterate over all frames."""
        return self.frames(skip=0)

    def frames(self, skip: int = config.FRAME_SKIP):
        """
        Generator that yields (frame, frame_index) tuples.

        Args:
            skip: Process every (skip+1)th frame.
                  0 = every frame. 1 = every other frame. etc.

        Yields:
            (frame: np.ndarray BGR, frame_idx: int)
        """
        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break  # End of video

            if frame_idx % (skip + 1) == 0:
                yield frame, frame_idx

            frame_idx += 1

    @property
    def metadata(self) -> dict:
        return {
            "path": self.path,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "duration_sec": round(self.frame_count / max(self.fps, 1), 2),
        }


class VideoWriter:
    """
    Writes annotated frames to an output MP4 file.

    Usage:
        with VideoWriter("out.mp4", fps=30, width=1280, height=720) as writer:
            for annotated_frame in frames:
                writer.write(annotated_frame)
    """

    def __init__(self, path: str, fps: float, width: int, height: int):
        self.path   = str(path)
        self.fps    = fps
        self.width  = width
        self.height = height
        self._writer = None
        self.frames_written = 0

        # Ensure output directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_CODEC)
        self._writer = cv2.VideoWriter(
            self.path, fourcc, self.fps, (self.width, self.height)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for: {self.path}")
        print(f"[VideoWriter] Writing to: {self.path}  ({self.width}×{self.height} @ {self.fps}fps)")
        return self

    def __exit__(self, *_):
        if self._writer:
            self._writer.release()
            print(f"[VideoWriter] Done. Wrote {self.frames_written} frames → {self.path}")

    def write(self, frame: np.ndarray) -> None:
        """
        Write a single BGR frame to the output video.
        Frame must match the width×height specified at construction.
        """
        # Resize if frame dimensions don't match (safety guard)
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        self._writer.write(frame)
        self.frames_written += 1
