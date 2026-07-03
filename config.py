import os
from dataclasses import dataclass, field


@dataclass
class YOLOConfig:
    model_name: str = "yolo11n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = "cpu"
    img_size: int = 640


@dataclass
class DangerConfig:
    proximity_px: int = 50
    overlap_iou: float = 0.01
    consecutive_frames_required: int = 3
    cooldown_seconds: float = 10.0


@dataclass
class IngestConfig:
    max_frame_size_bytes: int = 2_000_000
    target_fps: float = 3.0
    jpeg_quality: int = 80


@dataclass
class AppConfig:
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    danger: DangerConfig = field(default_factory=DangerConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    flagged_frames_dir: str = "data/flagged_frames"
    host: str = "0.0.0.0"
    port: int = 8000


def _from_env() -> AppConfig:
    cfg = AppConfig()
    if v := os.getenv("YOLO_MODEL"):
        cfg.yolo.model_name = v
    if v := os.getenv("YOLO_CONFIDENCE"):
        cfg.yolo.confidence_threshold = float(v)
    if v := os.getenv("YOLO_DEVICE"):
        cfg.yolo.device = v
    if v := os.getenv("YOLO_IMG_SIZE"):
        cfg.yolo.img_size = int(v)
    if v := os.getenv("DANGER_PROXIMITY_PX"):
        cfg.danger.proximity_px = int(v)
    if v := os.getenv("DANGER_CONSECUTIVE_FRAMES"):
        cfg.danger.consecutive_frames_required = int(v)
    if v := os.getenv("DANGER_COOLDOWN_SEC"):
        cfg.danger.cooldown_seconds = float(v)
    if v := os.getenv("SERVER_PORT"):
        cfg.port = int(v)
    return cfg


CONFIG = _from_env()
