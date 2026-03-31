"""
src/face_identifier.py
───────────────────────
Roster-free player identification using face embeddings + DBSCAN clustering.

═══════════════════════════════════════════════════════════════
THE HONEST PROBLEM STATEMENT
═══════════════════════════════════════════════════════════════

"Face-based player identification" means two very different things:

(A) "Tell me that #7 is MS Dhoni"
    → Requires a REFERENCE DATABASE of known faces.
    → Without it, no system in the world can do this reliably.

(B) "Consistently identify that the same person appears in frames
     1, 47, 203, and 891 — even if I don't know their name"
    → THIS is what we implement. It's called face re-identification.
    → Completely feasible without any roster.

We implement (B) and label players: Player-A, Player-B, etc.
If the user later tells us "#7 is Dhoni", we update the label.
This is the honest, robust, production-correct approach.

═══════════════════════════════════════════════════════════════
HOW IT WORKS — STEP BY STEP
═══════════════════════════════════════════════════════════════

STAGE 1 — Face detection (every FACE_DETECT_INTERVAL frames)
  InsightFace's RetinaFace model detects faces in the full frame.
  Returns: bounding box + 5 keypoints (eyes, nose, mouth corners).
  The keypoints are used for face alignment before embedding.

STAGE 2 — Face embedding (ArcFace, 512-dim vector)
  For each detected face, InsightFace runs ArcFace to produce a
  512-dimensional unit-norm embedding vector — the "face fingerprint".

  ArcFace is trained with additive angular margin loss:
    L = -log(e^(s·cos(θ+m)) / (e^(s·cos(θ+m)) + Σ e^(s·cos(θⱼ))))
  This pushes same-identity embeddings close together and
  different-identity embeddings far apart in angular space.

STAGE 3 — Track-face association
  We associate face detections with tracker IDs using IoU matching
  between the face bbox and the person bbox.
  Each track accumulates a gallery of face embeddings over time.

STAGE 4 — DBSCAN clustering (runs periodically)
  DBSCAN groups all accumulated face embeddings across all tracks.

  WHY DBSCAN not k-means?
  → We don't know how many players there are in advance.
  → DBSCAN discovers the number of clusters automatically.
  → It also marks noisy embeddings (bad-angle faces) as outliers.

  Distance metric: cosine distance = 1 - cosine_similarity(a, b)
  Same person: cosine distance ≈ 0.1–0.3
  Different:   cosine distance ≈ 0.6–1.4

STAGE 5 — Assign stable labels
  Each cluster gets a label: Player-A, Player-B, Player-C, ...
  These labels persist across the whole video.
  If a jersey number is detected for that track, we use "#7 Player-A".

═══════════════════════════════════════════════════════════════
CPU PERFORMANCE
═══════════════════════════════════════════════════════════════

buffalo_sc (what we use):
  - RetinaFace-mobilenet face detector: ~15ms/frame on CPU
  - ArcFace-mobilefacenet embedding:    ~8ms/face on CPU
  - Running every 10 frames: negligible overall overhead

buffalo_l (GPU-recommended):
  - RetinaFace-R50: ~100ms/frame → too slow for CPU
  - ArcFace-R100:   ~40ms/face  → fine for GPU

═══════════════════════════════════════════════════════════════
INTERVIEW QUESTIONS
═══════════════════════════════════════════════════════════════

Q: What is ArcFace and why is it better than FaceNet?
A: Both produce face embeddings. ArcFace uses additive angular margin
   in the loss function during training, which creates more
   discriminative embeddings (tighter intra-class, wider inter-class
   angular separation). State of the art on LFW, IJB-C benchmarks.

Q: Why DBSCAN over k-means for clustering faces?
A: k-means requires knowing k (number of players) in advance.
   DBSCAN discovers clusters density-based — it handles unknown k,
   marks outlier faces (side-angles, blurry) as noise, and doesn't
   assume spherical clusters (face embeddings are hyperspherical).

Q: What is cosine distance vs Euclidean distance for embeddings?
A: ArcFace produces unit-norm vectors (||v|| = 1), so they all lie on
   a hypersphere. Euclidean distance conflates angle and magnitude.
   Cosine similarity (dot product of unit vectors = cos of angle)
   measures pure directional similarity, which is what we want.
"""

import numpy as np
import cv2
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ── Try importing InsightFace ──────────────────────────────────────
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[FaceID] insightface not installed. Face identification disabled.")
    print("  Install: pip install insightface onnxruntime")

# ── DBSCAN for clustering ──────────────────────────────────────────
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[FaceID] scikit-learn not installed. Face clustering disabled.")


# Cluster index → readable label (A, B, C, ..., Z, AA, AB, ...)
def _cluster_label(idx: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if idx < 26:
        return f"Player-{letters[idx]}"
    return f"Player-{letters[idx // 26 - 1]}{letters[idx % 26]}"


class FaceIdentifier:
    """
    Detects faces, computes ArcFace embeddings, clusters them with DBSCAN,
    and assigns stable Player-A / Player-B labels to track IDs.
    """

    def __init__(self):
        self._app = None
        self._available = (
            INSIGHTFACE_AVAILABLE and SKLEARN_AVAILABLE and config.ENABLE_FACE_ID
        )

        # Embedding gallery: track_id → list of 512-dim np.ndarray
        self._gallery: dict[int, list] = defaultdict(list)

        # Best face crop per track (for the face grid output)
        self._best_crop: dict[int, np.ndarray] = {}
        self._best_crop_score: dict[int, float] = defaultdict(float)

        # Final labels: track_id → "Player-A" (or jersey-augmented)
        self.track_labels: dict[int, str] = {}

        # Cluster → label mapping (rebuilt on each clustering pass)
        self._cluster_to_label: dict[int, str] = {}

        if self._available:
            self._init_model()

    def _init_model(self):
        """Initialise InsightFace. Downloads models on first run (~10MB)."""
        print(f"[FaceID] Loading InsightFace ({config.FACE_MODEL_PACK})...")
        try:
            self._app = FaceAnalysis(
                name=config.FACE_MODEL_PACK,
                providers=["CPUExecutionProvider"],   # CPU-only, no CUDA needed
            )
            # det_size: face detection input size. 320 is faster than 640 on CPU.
            self._app.prepare(ctx_id=0, det_size=(320, 320))
            print("[FaceID] Ready.")
        except Exception as e:
            print(f"[FaceID] Init failed: {e}")
            self._available = False

    def process_frame(
        self,
        frame: np.ndarray,
        tracked_xyxy: np.ndarray,
        track_ids: np.ndarray,
    ) -> None:
        """
        Detect faces in the frame, match them to tracker IDs,
        store embeddings in the gallery.

        Args:
            frame:        Full BGR frame.
            tracked_xyxy: (N, 4) array of person bounding boxes [x1,y1,x2,y2].
            track_ids:    (N,) array of corresponding track IDs.
        """
        if not self._available or self._app is None:
            return
        if len(track_ids) == 0:
            return

        # InsightFace expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            faces = self._app.get(rgb)
        except Exception:
            return

        if not faces:
            return

        for face in faces:
            if face.embedding is None:
                continue

            # Face quality gate: use detection score as proxy for quality
            if face.det_score < 0.60:
                continue

            # Face bbox [x1,y1,x2,y2]
            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
            face_cx = (fx1 + fx2) / 2
            face_cy = (fy1 + fy2) / 2

            # Skip faces that are too small (likely background crowd)
            face_h = fy2 - fy1
            if face_h < config.MIN_FACE_SIZE:
                continue

            # Associate face with nearest person track (IoU matching)
            best_tid, best_iou = self._match_face_to_track(
                (fx1, fy1, fx2, fy2), tracked_xyxy, track_ids
            )
            if best_tid is None or best_iou < 0.05:
                continue

            # Normalise embedding to unit norm for cosine distance
            emb = face.embedding.astype(np.float32)
            emb /= (np.linalg.norm(emb) + 1e-8)
            self._gallery[int(best_tid)].append(emb)

            # Save face crop if it's the best quality seen for this track
            score = float(face.det_score) * face_h  # quality × size
            if score > self._best_crop_score[int(best_tid)]:
                self._best_crop_score[int(best_tid)] = score
                crop = frame[max(0, fy1):fy2, max(0, fx1):fx2]
                if crop.size > 0:
                    self._best_crop[int(best_tid)] = crop.copy()

    def cluster_and_label(self) -> None:
        """
        Run DBSCAN over all accumulated face embeddings to group
        track IDs that belong to the same person.

        Call this periodically (e.g. every 100 frames) rather than
        every frame — DBSCAN on 500+ embeddings takes ~50ms.
        """
        if not self._available or not SKLEARN_AVAILABLE:
            return

        # Collect all embeddings with their track IDs
        all_embs, all_tids = [], []
        for tid, embs in self._gallery.items():
            if not embs:
                continue
            # Use the mean embedding for each track (more stable than individual)
            mean_emb = np.mean(embs[-30:], axis=0)  # last 30 detections
            mean_emb /= (np.linalg.norm(mean_emb) + 1e-8)
            all_embs.append(mean_emb)
            all_tids.append(tid)

        if len(all_embs) < 2:
            # Not enough data for clustering yet
            for tid in all_tids:
                if tid not in self.track_labels:
                    self.track_labels[tid] = _cluster_label(len(self.track_labels))
            return

        X = np.stack(all_embs)  # shape: (n_tracks, 512)

        # DBSCAN with cosine metric
        # metric="cosine" interprets distance as 1 - cosine_similarity
        db = DBSCAN(
            eps=config.FACE_CLUSTER_EPS,
            min_samples=config.FACE_CLUSTER_MIN_SAMPLES,
            metric="cosine",
            n_jobs=-1,  # use all CPU cores
        ).fit(X)

        labels = db.labels_   # -1 = noise (unclassified face)
        unique_clusters = sorted(set(labels) - {-1})

        # Build cluster_idx → readable label mapping
        # Keep existing label assignments stable across re-clustering
        for cluster_idx in unique_clusters:
            if cluster_idx not in self._cluster_to_label:
                self._cluster_to_label[cluster_idx] = _cluster_label(
                    len(self._cluster_to_label)
                )

        # Assign labels to track IDs
        for i, (tid, cluster) in enumerate(zip(all_tids, labels)):
            if cluster == -1:
                # Noise point — assign a temporary unique label
                if tid not in self.track_labels:
                    self.track_labels[tid] = f"Player-?"
            else:
                self.track_labels[tid] = self._cluster_to_label[cluster]

    def get_label(self, track_id: int, jersey: str = "") -> str:
        """
        Get the display label for a track ID.

        Returns e.g.:
          "#7 Player-A"    if jersey OCR found #7 and face cluster is A
          "Player-A"       if face cluster found but no jersey
          "#7"             if jersey found but no face cluster yet
          ""               if nothing known yet (show just the track ID)
        """
        face_label = self.track_labels.get(int(track_id), "")
        jersey_key = str(jersey).strip() if jersey else ""
        roster_name = config.PLAYER_ROSTER.get(jersey_key, "") if jersey_key else ""

        if roster_name:
            return f"#{jersey_key} {roster_name}"
        elif jersey_key and face_label:
            return f"#{jersey_key} {face_label}"
        elif jersey_key:
            return f"#{jersey_key}"
        elif face_label:
            return face_label
        return ""

    def save_face_grid(self, output_path: str = config.OUTPUT_FACE_GRID) -> None:
        """
        Saves a grid image of the best face crop per cluster.
        Useful for visually verifying clustering quality.
        """
        if not self._best_crop or not config.SAVE_FACE_CROPS:
            return

        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # Group by cluster label
        label_to_crops: dict[str, list] = defaultdict(list)
        for tid, crop in self._best_crop.items():
            label = self.track_labels.get(tid, f"ID-{tid}")
            label_to_crops[label].append(crop)

        n = len(label_to_crops)
        if n == 0:
            return

        cols = min(n, 5)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.4))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        for idx, (label, crops) in enumerate(sorted(label_to_crops.items())):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            # Use the first (best-quality) crop
            crop = crops[0]
            # Resize to uniform thumbnail
            thumb = cv2.resize(crop, (80, 100))
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            ax.imshow(thumb_rgb)
            ax.set_title(label, fontsize=8, fontweight="bold")
            ax.axis("off")

        # Hide unused axes
        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].axis("off")

        plt.suptitle("Detected players (face clusters)", fontsize=11, fontweight="bold")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[FaceID] Face grid saved → {output_path} ({n} identities)")

    @staticmethod
    def _match_face_to_track(
        face_bbox: tuple,
        tracked_xyxy: np.ndarray,
        track_ids: np.ndarray,
    ) -> tuple:
        """
        Match a face bbox to the best-overlapping person track via IoU.
        Returns (track_id, iou) of the best match, or (None, 0).
        """
        if len(tracked_xyxy) == 0:
            return None, 0.0

        fx1, fy1, fx2, fy2 = face_bbox
        fa = max(0, (fx2 - fx1) * (fy2 - fy1))
        if fa == 0:
            return None, 0.0

        best_tid, best_iou = None, 0.0
        for i, (px1, py1, px2, py2) in enumerate(tracked_xyxy):
            ix1, iy1 = max(fx1, px1), max(fy1, py1)
            ix2, iy2 = min(fx2, px2), min(fy2, py2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            pa  = max(0, (px2 - px1) * (py2 - py1))
            iou = inter / (fa + pa - inter + 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_tid = int(track_ids[i])

        return best_tid, best_iou

    @property
    def is_available(self) -> bool:
        return self._available

    def summary(self) -> dict:
        return {
            "unique_face_clusters": len(set(self.track_labels.values()) - {"Player-?"}),
            "tracks_with_face":     len(self._gallery),
            "total_embeddings":     sum(len(v) for v in self._gallery.values()),
        }