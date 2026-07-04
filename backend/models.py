from enum import Enum
from pydantic import BaseModel


class DetectionOut(BaseModel):
    class_id: int
    class_name: str
    category: str
    box: list[int]  # [x1, y1, x2, y2]
    confidence: float


class AlertSeverity(str, Enum):
    WARNING = "WARNING"
    DANGER = "DANGER"
    OK = "OK"


class AlertOut(BaseModel):
    id: str
    rule_name: str
    severity: AlertSeverity
    person: DetectionOut
    hazard: DetectionOut
    distance_px: float
    overlap_iou: float
    timestamp: float
    frame_thumbnail_url: str | None = None


class PPECheckOut(BaseModel):
    id: str
    severity: AlertSeverity          # DANGER (missing something) or OK
    missing: list[str]               # subset of ["hardhat", "vest"]
    has_hardhat: bool
    has_vest: bool
    person: DetectionOut
    timestamp: float
    frame_thumbnail_url: str | None = None


class FrameResultOut(BaseModel):
    frame_id: int
    timestamp: float
    mode: str = "site"               # "site" or "checkpoint"
    detections: list[DetectionOut]
    active_dangers: list[AlertOut] = []
    confirmed_alerts: list[AlertOut] = []
    ppe_checks: list[PPECheckOut] = []
    frame_jpeg_b64: str
    processing_ms: float


class StatsOut(BaseModel):
    total_frames_processed: int
    total_alerts: int
    uptime_seconds: float
    current_fps: float
