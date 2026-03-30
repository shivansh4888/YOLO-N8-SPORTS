"""
utils/heatmap.py
────────────────
Generates a movement heatmap from all tracked player positions.

WHAT IS A HEATMAP?
  A 2D histogram of where players spent time on the field.
  Hot spots = high traffic zones. Used by sports analysts to see:
  • Which areas of the pitch are most contested.
  • Player positioning tendencies.
  • Team formation patterns.

HOW IT WORKS:
  1. Collect ALL (cx, cy) centre-points across ALL frames and ALL IDs.
  2. Build a 2D histogram (bin the positions into a grid).
  3. Apply Gaussian blur — smooths the discrete bins into a heatmap.
  4. Normalise to 0–255 and apply a colour map (hot/jet/plasma).
  5. Overlay on a reference frame or blank canvas, save as PNG.

INTERVIEW QUESTION:
Q: Why Gaussian blur the histogram?
A: Raw histograms are blocky (staircase artifacts). Gaussian blur
   spreads each data point to nearby bins, creating smooth contours
   that are visually meaningful and statistically defensible —
   it's a kernel density estimate on a 2D grid.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def generate_heatmap(
    positions: list[tuple[float, float]],
    frame_shape: tuple[int, int],          # (height, width)
    reference_frame: np.ndarray = None,
    output_path: str = config.OUTPUT_HEATMAP,
) -> None:
    """
    Generate and save a movement heatmap image.

    Args:
        positions:       List of (cx, cy) pixel coordinates from tracker.
        frame_shape:     (H, W) of the video frame.
        reference_frame: Optional BGR frame to overlay the heatmap on.
                         If None, uses a black background.
        output_path:     Where to save the PNG.
    """
    if not positions:
        print("[Heatmap] No positions to plot. Skipping.")
        return

    h, w = frame_shape

    # ── Build 2D histogram ────────────────────────────────────────
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])

    # np.histogram2d: bin the scatter into a grid
    # Note: histogram2d uses (x, y) but image uses (row=y, col=x)
    heatmap, xedges, yedges = np.histogram2d(
        xs, ys,
        bins=[w // 4, h // 4],     # bin resolution: every 4 pixels
        range=[[0, w], [0, h]],
    )

    # Transpose so that rows=y, cols=x (image convention)
    heatmap = heatmap.T

    # ── Gaussian smoothing ────────────────────────────────────────
    heatmap = gaussian_filter(heatmap, sigma=config.HEATMAP_SIGMA)

    # ── Normalise to 0–255 ────────────────────────────────────────
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)

    # ── Resize to match frame dimensions ─────────────────────────
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # ── Apply colour map ──────────────────────────────────────────
    colormap = {
        "hot":    cv2.COLORMAP_HOT,
        "jet":    cv2.COLORMAP_JET,
        "plasma": cv2.COLORMAP_PLASMA,
    }.get(config.HEATMAP_COLORMAP, cv2.COLORMAP_HOT)

    heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)

    # ── Overlay on reference frame or black background ────────────
    if reference_frame is not None:
        # Blend: 40% heatmap + 60% original frame
        bg = cv2.resize(reference_frame, (w, h))
        output_img = cv2.addWeighted(bg, 0.6, heatmap_colored, 0.4, 0)
    else:
        output_img = heatmap_colored

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, output_img)
    print(f"[Heatmap] Saved → {output_path}  ({len(positions)} positions plotted)")
