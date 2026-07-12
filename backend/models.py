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


class MarkerDetectionOut(BaseModel):
    marker_id: int
    corners: list[list[float]]       # 4 corners as [[x, y], ...]
    center: list[float]              # [cx, cy]


class PersonDistanceOut(BaseModel):
    """Distance of a detected person from the calibrated reference area,
    measured in metres. `distance_m` is signed: negative → inside area."""
    person_box: list[int]            # [x1, y1, x2, y2]
    distance_m: float
    inside: bool


class CalibrationOut(BaseModel):
    camera_id: str
    marker_ids: list[int]
    width_m: float
    height_m: float
    created_at: float


class ActiveZoneOut(BaseModel):
    """Zone with its polygon resolved for the current frame.

    Sent per frame over WebSocket so frontends can draw the polygon
    even for marker-defined zones (whose stored polygon is empty and
    only exists at runtime from live ArUco detections)."""
    id: str
    name: str
    severity: str
    polygon: list[list[float]]
    marker_ids: list[int] = []


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
    markers: list[MarkerDetectionOut] = []
    person_distances: list[PersonDistanceOut] = []
    active_zones: list[ActiveZoneOut] = []
    calibration_active: bool = False
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
