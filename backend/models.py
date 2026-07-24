import math
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    distance_m: float | None = None
    calibrated: bool = False
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


class PostureAssessmentOut(BaseModel):
    id: str
    track_id: int
    severity: AlertSeverity
    status: str
    risk_score: float
    signals: list[str]
    metrics: dict[str, float]
    pose_confidence: float
    history_seconds: float
    person: DetectionOut
    # MediaPipe Pose topology in normalized image coordinates:
    # [x, y, z, visibility].  The panel draws this locally so the live
    # preview can remain an unannotated frame and its layers stay toggleable.
    pose_landmarks: list[list[float]] = []
    timestamp: float
    confirmed: bool = False
    frame_thumbnail_url: str | None = None

class ZoneBreachOut(BaseModel):
    id: str
    zone_id: str
    zone_name: str
    severity: AlertSeverity
    rule_name: str = "zone_breach"
    inside: bool = True
    distance_px: float | None = None
    distance_m: float | None = None
    calibrated: bool = False
    person: DetectionOut
    timestamp: float
    frame_thumbnail_url: str | None = None


class MarkerDetectionOut(BaseModel):
    marker_id: int
    corners: list[list[float]]       # 4 corners as [[x, y], ...]
    center: list[float]              # [cx, cy]


class DynamicSafetyZoneOut(BaseModel):
    zone_id: str
    severity: AlertSeverity
    hazard: DetectionOut
    polygon: list[list[float]]       # normalized [[x, y], ...]
    threshold_m: float | None = None
    threshold_px: float | None = None
    calibrated: bool = False


class WorkerIdentificationOut(BaseModel):
    worker_id: str
    source: str = "qr"
    person_box: list[int]
    tag_polygon: list[list[float]] = []
    cached: bool = False
    registered: bool = False
    first_name: str | None = None
    last_name: str | None = None
    full_name: str | None = None
    position: str | None = None
    department: str | None = None


class WorkerProfileFields(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    first_name: str = Field(min_length=1, max_length=120)
    last_name: str = Field(min_length=1, max_length=120)
    position: str = Field(min_length=1, max_length=160)
    department: str = Field(min_length=1, max_length=160)

    @field_validator("first_name", "last_name", "position", "department")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        if any(ord(char) < 32 for char in value):
            raise ValueError("control characters are not allowed")
        return value


class WorkerCreateIn(WorkerProfileFields):
    worker_id: str = Field(min_length=1, max_length=64)

    @field_validator("worker_id")
    @classmethod
    def validate_worker_id(cls, value: str) -> str:
        if any(ord(char) < 32 for char in value):
            raise ValueError("control characters are not allowed")
        return value


class WorkerUpdateIn(WorkerProfileFields):
    pass


class WorkerProfileOut(WorkerCreateIn):
    full_name: str
    created_at: float
    updated_at: float


class UnidentifiedWorkerOut(BaseModel):
    id: str
    severity: AlertSeverity = AlertSeverity.WARNING
    person: DetectionOut
    timestamp: float
    frame_thumbnail_url: str | None = None


class PersonDistanceOut(BaseModel):
    """Nearest signed distance from a person to a configured static zone."""
    person_box: list[int]            # [x1, y1, x2, y2]
    zone_id: str
    zone_name: str
    distance_px: float | None = None
    distance_m: float | None = None
    inside: bool
    calibrated: bool = False


class CalibrationQualityOut(BaseModel):
    valid: bool = True
    status: str
    score: float = 0.0
    score_percent: float = 0.0
    coverage_ratio: float = 0.0
    quadrilateral_area_ratio: float = 0.0
    min_edge_ratio: float = 0.0
    max_edge_ratio: float = 0.0
    smallest_marker_area_px: float = 0.0
    min_angle_deg: float = 0.0
    max_angle_deg: float = 0.0
    reprojection_error_normalized: float | None = None
    reprojection_error_m: float | None = None
    condition_number: float | None = None
    warnings: list[str] = []
    messages: list[str] = []


class CalibrationOut(BaseModel):
    camera_id: str
    marker_ids: list[int]
    width_m: float
    height_m: float
    # Homography input is normalized [u, v], not source-frame pixels.
    homography: list[list[float]]
    source_frame_width: int
    source_frame_height: int
    source_aspect_ratio: float
    # Both center arrays follow marker_ids order: TL, TR, BR, BL.
    marker_centers: list[list[float]]
    marker_centers_px: list[list[float]]
    quality: CalibrationQualityOut
    # Explicit spec name; `quality` remains for backward compatibility.
    calibration_quality: CalibrationQualityOut
    created_at: float


class CalibrationFromLatestIn(BaseModel):
    marker_ids: list[int] = Field(min_length=4, max_length=4)
    width_m: float = Field(gt=0)
    height_m: float = Field(gt=0)

    @field_validator("marker_ids")
    @classmethod
    def validate_calibration_marker_ids(cls, value: list[int]) -> list[int]:
        if len(set(value)) != 4:
            raise ValueError("marker_ids must contain four unique IDs")
        invalid = [marker_id for marker_id in value if marker_id < 0 or marker_id >= 50]
        if invalid:
            raise ValueError(f"marker IDs outside DICT_4X4_50 range 0..49: {invalid}")
        return value

    @field_validator("marker_ids", mode="before")
    @classmethod
    def parse_calibration_marker_ids(cls, value):
        if isinstance(value, str):
            try:
                return [int(part.strip()) for part in value.split(",") if part.strip()]
            except ValueError as exc:
                raise ValueError("marker_ids must be ints or a comma-separated string") from exc
        return value


class CalibrationMeasureIn(BaseModel):
    point_a: list[float] = Field(min_length=2, max_length=2)
    point_b: list[float] = Field(min_length=2, max_length=2)
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)

    @field_validator("point_a", "point_b")
    @classmethod
    def validate_measure_point(cls, value: list[float]) -> list[float]:
        if not all(math.isfinite(float(coordinate)) for coordinate in value):
            raise ValueError("point coordinates must be finite")
        return [float(coordinate) for coordinate in value]


class CalibrationMeasureOut(BaseModel):
    point_a_m: list[float]
    point_b_m: list[float]
    distance_m: float


class CalibrationPreviewMarkerOut(BaseModel):
    marker_id: int
    center: list[float]
    center_normalized: list[float]
    corners: list[list[float]]


class CalibrationPreviewOut(BaseModel):
    camera_id: str
    frame_available: bool
    frame_age_sec: float | None = None
    timestamp: float | None = None
    received_at: float | None = None
    mode: str | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    frame_aspect: float | None = None
    fresh: bool = False
    required_marker_ids: list[int]
    detected_markers: list[CalibrationPreviewMarkerOut]
    missing_marker_ids: list[int]
    valid: bool
    validation_error: str | None = None
    quality: CalibrationQualityOut | None = None
    calibration_quality: CalibrationQualityOut | None = None
    calibration_exists: bool = False
    calibration_valid_for_frame: bool = False
    # Compatibility aliases kept for older clients during the UI migration.
    calibration_active: bool = False
    calibration_aspect_compatible: bool = False
    calibration_compatibility_warning: str | None = None
    calibration: CalibrationOut | None = None


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
    warning_distance_m: float = 1.5
    warning_distance_px: float = 60.0


class FrameResultOut(BaseModel):
    frame_id: int
    timestamp: float
    camera_id: str = "cam_default"
    mode: str = "site"               # "site" or "checkpoint"
    detections: list[DetectionOut]
    active_dangers: list[AlertOut] = []
    confirmed_alerts: list[AlertOut] = []
    ppe_checks: list[PPECheckOut] = []
    posture_assessments: list[PostureAssessmentOut] = []
    confirmed_posture_alerts: list[PostureAssessmentOut] = []
    posture_available: bool = False
    active_zone_breaches: list[ZoneBreachOut] = []
    confirmed_zone_breaches: list[ZoneBreachOut] = []
    markers: list[MarkerDetectionOut] = []
    worker_identifications: list[WorkerIdentificationOut] = []
    unidentified_workers: list[UnidentifiedWorkerOut] = []
    worker_identification_available: bool = False
    dynamic_safety_zones: list[DynamicSafetyZoneOut] = []
    person_distances: list[PersonDistanceOut] = []
    active_zones: list[ActiveZoneOut] = []
    calibration_active: bool = False
    calibration_valid: bool = False
    calibration_warning: str | None = None
    # Unannotated camera frame used only as the live-view background.
    # Evidence thumbnails/clips remain rendered from the annotated server
    # frame and are never affected by an operator's local layer switches.
    frame_jpeg_b64: str
    processing_ms: float


class FrameAcceptedOut(BaseModel):
    """Immediate acknowledgement for background ``latest-wins`` ingest."""

    accepted: bool = True
    camera_id: str
    timestamp: float
    mode: str
    queue_depth: int
    replaced_pending: bool = False
    dropped_frames: int = 0


class AlarmRecord(BaseModel):
    """Unified persistent record for both site hazards and PPE violations."""
    id: str
    timestamp: float
    mode: str                        # "site" or "checkpoint"
    kind: str                        # site_hazard | ppe_missing | zone_breach | posture_anomaly | fall_detected | smoking_gesture
    severity: AlertSeverity
    rule_name: str
    description: str                 # human-readable, e.g. "Osoba w strefie pojazdu"
    camera_id: str = "cam_default"
    thumbnail_url: str | None = None
    clip_url: str | None = None
    review_status: str = "new"       # new | acknowledged | confirmed | false_positive | escalated
    reviewed_at: float | None = None
    reviewed_by: str | None = None
    review_note: str | None = None
    details: dict = {}


class StatsOut(BaseModel):
    total_frames_processed: int
    total_alerts: int
    uptime_seconds: float
    current_fps: float
