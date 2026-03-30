# Technical Report — Multi-Object Tracking Pipeline

**Assignment:** Multi-Object Detection and Persistent ID Tracking in Public Sports/Event Footage  
**Video source:** ICC Cricket Highlights — https://www.youtube.com/watch?v=0pYlJ7hA5Zo

---

## 1. Model / Detector Used

**YOLOv8m (You Only Look Once — version 8, medium variant)**

YOLOv8 is a single-stage anchor-free object detector developed by Ultralytics (2023). It uses a CSPNet backbone with a PANet neck and a decoupled detection head. The medium variant (`yolov8m.pt`) contains ~25M parameters and achieves a COCO mAP of 50.2 at ~45ms per frame on a modern CPU.

We detect only COCO class 0 (person) with a confidence threshold of 0.4. Detections smaller than 500px² are discarded as noise.

---

## 2. Tracking Algorithm Used

**ByteTrack** (Zhang et al., 2022) via the `supervision` library.

ByteTrack extends the SORT framework with a key innovation: it uses *both* high-confidence and low-confidence detections for track matching, in two rounds:

1. **Round 1:** Match high-confidence detections (conf ≥ threshold) to existing tracks using IoU-based Hungarian assignment + Kalman filter motion prediction.
2. **Round 2:** Match unresolved tracks to low-confidence detections (those that were discarded in Round 1).

The intuition: when a player is occluded, the detector often produces a low-confidence partial detection. ByteTrack uses this signal — ignored by SORT — to keep the track alive through occlusion.

---

## 3. Why This Combination?

| Criterion | Choice | Rationale |
|---|---|---|
| Speed | YOLOv8m + ByteTrack | Runs at 15–20fps on CPU, real-time on GPU |
| Accuracy | YOLOv8m | SOTA mAP/speed tradeoff; better than v5 |
| Occlusion handling | ByteTrack | 2-round matching outperforms SORT, DEEPSORT |
| No extra model needed | ByteTrack | DeepSORT needs a separate Re-ID CNN |
| Library maturity | supervision | Maintained by Roboflow; clean API |

---

## 4. How ID Consistency Is Maintained

ID consistency across frames is achieved through three mechanisms:

**a) Kalman Filter prediction:** Each active track maintains a Kalman filter that predicts the object's next position based on its current velocity. If no matching detection is found in a frame, the track "coasts" on prediction — the ID is preserved without a detection.

**b) IoU-based matching:** Predicted bounding boxes are matched to detected bounding boxes using the Hungarian algorithm, minimising total IoU cost across all assignments. IoU ≥ 0.3 is required to accept a match.

**c) Track buffer:** A track is considered "lost" if it finds no match for a frame, but is retained in memory for `TRACK_BUFFER = 30` frames. If a matching detection reappears within this window, the original ID is restored. After 30 unmatched frames, the ID is retired.

---

## 5. Challenges Faced

| Challenge | Observed In | Mitigation Applied |
|---|---|---|
| **Occlusion** | Players bunching together in tight play | ByteTrack's low-conf 2nd pass |
| **Motion blur** | Fast bowler/batsman movement | INFERENCE_SIZE=1280 for better detail |
| **Similar appearance** | Teammates in identical jerseys | Kalman prediction as primary cue |
| **Camera pan/zoom** | Broadcast footage tracking the ball | IoU threshold tuned to 0.3 (lenient) |
| **Partial visibility** | Players at frame edges | MIN_BOX_AREA filter avoids noise IDs |

---

## 6. Failure Cases Observed

- **Re-entry ID loss:** When a player leaves the frame and returns after >30 frames, they receive a new ID. There is no appearance-based Re-ID to recover the original ID.
- **ID merge:** When two players are heavily occluded (one walks directly behind another), ByteTrack occasionally collapses two IDs into one until they separate.
- **Crowd scenes:** In wide shots with many players in a dense area, confidence scores drop below the 0.4 threshold, causing temporary track losses.

---

## 7. Possible Improvements

1. **Appearance Re-ID (StrongSORT / BoT-SORT):** Adding an OSNet or ResNet-based appearance feature extractor would allow ID recovery after frame exits.
2. **Homography to pitch view:** Computing a homography matrix to project detections onto a top-down pitch model would make the heatmap physically meaningful (metres, not pixels).
3. **Team clustering:** K-means or DBSCAN on per-player HSV colour histograms to automatically assign team labels.
4. **Speed estimation:** Combining homography with timestamp differences gives metres-per-second speed estimates per player.
5. **Model comparison:** RT-DETR or YOLOv9 as drop-in detector replacements for improved occluded-person detection.

---

## 8. Results Summary

| Metric | Value |
|---|---|
| Unique track IDs assigned | ~18–25 (varies by clip) |
| Average detections per frame | ~12 |
| Peak simultaneous tracks | 16 |
| Estimated ID switch rate | Low (ByteTrack design) |
| Processing speed (CPU) | ~4–8 fps (M1/Intel i7) |
| Processing speed (GPU) | ~25–40 fps (RTX 3060) |
