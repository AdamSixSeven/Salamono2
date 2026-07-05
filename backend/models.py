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


class ZoneBreachOut(BaseModel):
    id: str
    zone_id: str
    zone_name: str
    severity: AlertSeverity
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
    active_zone_breaches: list[ZoneBreachOut] = []
    confirmed_zone_breaches: list[ZoneBreachOut] = []
    frame_jpeg_b64: str
    processing_ms: float


class AlarmRecord(BaseModel):
    """Unified persistent record for both site hazards and PPE violations."""
    id: str
    timestamp: float
    mode: str                        # "site" or "checkpoint"
    kind: str                        # "site_hazard" | "ppe_missing"
    severity: AlertSeverity
    rule_name: str
    description: str                 # human-readable, e.g. "Osoba w strefie pojazdu"
    camera_id: str = "cam_default"
    thumbnail_url: str | None = None
    details: dict = {}


class StatsOut(BaseModel):
    total_frames_processed: int
    total_alerts: int
    uptime_seconds: float
    current_fps: float
