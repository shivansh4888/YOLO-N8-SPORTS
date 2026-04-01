FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --upgrade pip

RUN pip install \
    flask \
    gunicorn \
    yt-dlp \
    ultralytics \
    opencv-python-headless \
    supervision

# Pre-download YOLO model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

ENV PORT=8080

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--timeout", "600"]