FROM python:3.10-slim

# Install system deps (IMPORTANT)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Install python deps
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    flask \
    gunicorn \
    yt-dlp \
    ultralytics \
    easyocr \
    insightface \
    onnxruntime \
    scikit-learn \
    opencv-python-headless \
    supervision

# Expose port
ENV PORT=8080

# Start server
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--timeout", "600"]
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"