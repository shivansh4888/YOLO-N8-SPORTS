"""
download_video.py
─────────────────
Downloads a publicly available video using yt-dlp.

HOW TO USE:
    python download_video.py --url "https://www.youtube.com/watch?v=XXXX"
    python download_video.py  # uses the default URL below

The video is saved to data/input_video.mp4

WHY yt-dlp (not youtube-dl)?
  yt-dlp is the actively maintained fork with faster updates,
  better format selection, and SponsorBlock support.
  youtube-dl was effectively abandoned in 2021.

NOTE ON VIDEO SELECTION:
  We use a publicly available sports broadcast clip.
  For cricket:  search "cricket highlights" on YouTube.
  For football: search "football match highlights full game".
  Ideal length: 30–90 seconds for fast iteration during development.

RECOMMENDED TEST VIDEO:
  FIFA World Cup public highlights clips — multiple clearly visible
  players, camera panning, occlusion events, good lighting variety.
"""

import subprocess
import sys
import os

# ── Default public video URL ──────────────────────────────────────
# This is a publicly available cricket/football highlights clip.
# You can override it with --url argument or by changing this line.
DEFAULT_URL = "https://www.youtube.com/watch?v=0pYlJ7hA5Zo"  # ICC cricket highlights

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "input_video.mp4")


def download_video(url: str, output_path: str) -> bool:
    """
    Download a video from a public URL using yt-dlp.

    Args:
        url:         Public video URL (YouTube, Vimeo, etc.)
        output_path: Where to save the MP4 file.

    Returns:
        True if successful, False otherwise.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # yt-dlp command:
    #   -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]"
    #     → best quality MP4 that includes both video and audio
    #   --merge-output-format mp4
    #     → ensure final container is .mp4
    #   -o output_path
    #     → save to our specified path
    #   --no-playlist
    #     → only download the single video, not a whole playlist

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[ext=mp4]",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "-o", output_path,
        url,
    ]

    print(f"[Downloader] Fetching: {url}")
    print(f"[Downloader] Saving to: {output_path}")
    print(f"[Downloader] Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"\n[Downloader] Success! Saved to: {output_path}")
        return True
    else:
        print(f"\n[Downloader] ERROR — yt-dlp exited with code {result.returncode}")
        print("  Make sure yt-dlp is installed: pip install yt-dlp")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download a public sports video.")
    parser.add_argument(
        "--url", type=str, default=DEFAULT_URL,
        help="Public video URL to download",
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_PATH,
        help="Output path for the downloaded video",
    )
    args = parser.parse_args()

    success = download_video(args.url, args.output)
    sys.exit(0 if success else 1)
