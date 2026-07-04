FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO model so container starts instantly
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

COPY config.py .
COPY backend/ backend/
COPY frontend/ frontend/
COPY phone/ phone/
COPY etap0/ etap0/
COPY tools/ tools/

RUN mkdir -p data/flagged_frames data/sample_videos

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
