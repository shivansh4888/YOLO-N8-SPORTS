# ── Base image ────────────────────────────────────────────────────
# python:3.11-slim is ~150MB vs full python:3.11 (~1GB)
FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────
# ffmpeg  → video re-encoding for browser playback
# libgl1  → OpenCV needs it on headless servers
# yt-dlp  → YouTube downloader (installed via pip, updated regularly)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────
# Copy requirements first — Docker layer caching means if requirements
# don't change, pip install is skipped on re-builds. Huge time saver.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── EasyOCR model pre-download ────────────────────────────────────
# Download language models at build time so the first request isn't slow.
# The models are cached at /root/.EasyOCR/
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)" || true

# ── YOLO model pre-download ───────────────────────────────────────
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')" || true

# ── Copy application code ─────────────────────────────────────────
COPY . .

# ── Create required directories ───────────────────────────────────
RUN mkdir -p data outputs

# ── Port (Railway injects $PORT at runtime) ───────────────────────
EXPOSE 5000

# ── Start command ─────────────────────────────────────────────────
# gunicorn: production WSGI server (not Flask's dev server)
# --timeout 600: video processing can take several minutes
# --workers 1: each worker needs full RAM for YOLO; 1 is correct for free tier
CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:$PORT", \
     "--timeout", "600", \
     "--workers", "1", \
     "--threads", "4", \
     "--worker-class", "gthread", \
     "--log-level", "info"]