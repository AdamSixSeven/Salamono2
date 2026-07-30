FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO model so container starts instantly
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Pre-download PPE and MediaPipe posture models.
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -L -o ppe.pt "https://huggingface.co/Hansung-Cho/yolov8-ppe-detection/resolve/main/best.pt" && \
    mkdir -p models && \
    curl -L -o models/pose_landmarker_heavy.task \
      "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_heavy.task" && \
    apt-get remove -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY config.py .
COPY backend/ backend/
COPY pose_behavior/ pose_behavior/
COPY frontend/ frontend/
COPY phone/ phone/
COPY tools/ tools/
COPY models/ models/

RUN mkdir -p data/flagged_frames data/event_clips data/sample_videos

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
