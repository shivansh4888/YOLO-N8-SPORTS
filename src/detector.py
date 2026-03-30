"""
src/detector.py
───────────────
YOLOv8-based object detector — the first stage of the pipeline.

WHAT THIS FILE DOES:
  1. Loads a pretrained YOLOv8 model (downloads weights automatically).
  2. Runs inference on a single video frame.
  3. Returns detections as a clean numpy array of bounding boxes.

THE DETECT → TRACK SEPARATION (key interview concept):
  Detection  = "where are the people in THIS frame?"
  Tracking   = "which person in frame N is the same as in frame N-1?"
  These are TWO separate problems. Good engineers keep them separate.

WHAT IS YOLO?
  You Only Look Once — a single-pass CNN that divides the image into
  a grid and predicts bounding boxes + class probabilities per cell.
  YOLOv8 (2023) is the current standard: ~50ms per frame on CPU.

BOUNDING BOX FORMAT — xyxy:
  [x1, y1, x2, y2] where (x1,y1) = top-left, (x2,y2) = bottom-right
  All coordinates in pixels relative to the original frame size.

INTERVIEW QUESTIONS:
Q: What is Non-Maximum Suppression (NMS)?
A: After YOLO predicts many overlapping boxes for one object,
   NMS keeps the highest-confidence box and removes others with
   IoU > threshold. Result: one clean box per object.

Q: Why filter by MIN_BOX_AREA?
A: Tiny boxes are usually noise — distant crowd, camera artifacts.
   Filtering keeps the tracker from wasting IDs on junk detections.

Q: What's the difference between yolov8n and yolov8x?
A: nano = 3.2M params, ~160fps on GPU, less accurate.
   extra = 68M params, ~30fps, most accurate. 'm' is the sweet spot.
"""

from ultralytics import YOLO
import numpy as np
import sys, os

# Add project root to path so we can import config from anywhere
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class Detector:
    """
    Wraps YOLOv8 and exposes a single clean method: detect(frame).

    Why a class and not just a function?
    → The model is expensive to load (~300ms). Wrapping it in a class
      means we load it once at construction and reuse it every frame.
    """

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL,
        confidence: float = config.CONFIDENCE_THRESHOLD,
        target_classes: list = config.TARGET_CLASSES,
        min_box_area: int = config.MIN_BOX_AREA,
    ):
        """
        Args:
            model_path:     Path or name of YOLO weights.
                            If just a name like 'yolov8m.pt', Ultralytics
                            auto-downloads from their model hub.
            confidence:     Minimum detection confidence (0–1).
            target_classes: List of COCO class IDs to keep.
                            0 = person. None = keep all.
            min_box_area:   Discard detections smaller than this (px²).
        """
        print(f"[Detector] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.target_classes = target_classes
        self.min_box_area = min_box_area
        print(f"[Detector] Ready. Conf={confidence}, Classes={target_classes}")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run YOLOv8 inference on a single BGR frame (as returned by OpenCV).

        Args:
            frame: np.ndarray of shape (H, W, 3), dtype uint8, BGR color.

        Returns:
            detections: np.ndarray of shape (N, 6)
                        Each row = [x1, y1, x2, y2, confidence, class_id]
                        N = number of valid detections (can be 0).

        WHY RETURN RAW NUMPY ARRAYS?
        → Framework-agnostic. The tracker, annotator, and stats modules
          don't need to know about YOLO internals — they just get numbers.
          This is the "single responsibility principle" in practice.
        """
        # Run YOLO inference
        # verbose=False silences the per-frame print spam
        results = self.model(
            frame,
            conf=self.confidence,
            classes=self.target_classes,
            verbose=False,
            imgsz=config.INFERENCE_SIZE,
        )[0]  # [0] because YOLO returns a list (one result per image)

        # ── Extract boxes ─────────────────────────────────────────
        # results.boxes.data is a tensor of shape (N, 6):
        #   [x1, y1, x2, y2, confidence, class_id]
        if results.boxes is None or len(results.boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        # Move to CPU and convert to numpy
        detections = results.boxes.data.cpu().numpy().astype(np.float32)

        # ── Filter by minimum bounding box area ───────────────────
        # Area = width × height = (x2-x1) × (y2-y1)
        widths  = detections[:, 2] - detections[:, 0]
        heights = detections[:, 3] - detections[:, 1]
        areas   = widths * heights

        # Keep only detections whose area exceeds the threshold
        mask = areas >= self.min_box_area
        detections = detections[mask]

        return detections  # shape: (N, 6)

    def get_model_info(self) -> dict:
        """
        Returns a summary dict — useful for logging and the technical report.
        """
        return {
            "model": config.YOLO_MODEL,
            "confidence_threshold": self.confidence,
            "target_classes": self.target_classes,
            "min_box_area": self.min_box_area,
            "inference_size": config.INFERENCE_SIZE,
        }
