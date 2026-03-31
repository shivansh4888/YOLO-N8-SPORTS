"""
utils/pick_points.py
─────────────────────
Click 4 pitch corners on the first video frame.
Prints pixel coordinates to copy into config.py PITCH_PIXEL_PTS.

Usage:  python utils/pick_points.py
"""
import cv2, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

pts = []

def click(event, x, y, flags, img):
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append((x, y))
        cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(img, str(len(pts)), (x+8, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Click 4 corners — Q to quit", img)
        if len(pts) == 4:
            print("\n=== Copy into config.py ===")
            print(f"PITCH_PIXEL_PTS = {pts}")
            print("===========================\n")

cap = cv2.VideoCapture(config.INPUT_VIDEO)
ret, frame = cap.read(); cap.release()
if not ret:
    print("ERROR: could not read video."); sys.exit(1)

disp = frame.copy()
cv2.namedWindow("Click 4 corners — Q to quit")
cv2.setMouseCallback("Click 4 corners — Q to quit", click, disp)
print("Click: top-left, top-right, bottom-right, bottom-left of pitch rectangle")
print("Real-world pts:", config.PITCH_REAL_PTS)
while True:
    cv2.imshow("Click 4 corners — Q to quit", disp)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cv2.destroyAllWindows()
