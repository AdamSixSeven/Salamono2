import os
from dataclasses import dataclass, field


@dataclass
class YOLOConfig:
    model_name: str = "yolo11n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = "cpu"
    # 512 is the balanced live-preview default.  Raise to 640 when small or
    # distant people are more important than latency.
    img_size: int = 512
    # COCO defaults: person=0, car=2, bus=5, truck=7.  For a custom
    # construction-equipment checkpoint set YOLO_HAZARD_CLASS_IDS to the
    # model-specific class IDs, e.g. "1,3,6".
    person_class_ids: list[int] = field(default_factory=lambda: [0])
    hazard_class_ids: list[int] = field(default_factory=lambda: [2, 5, 7])


@dataclass
class DangerConfig:
    # Fallback image-space thresholds used when no ground-plane calibration
    # exists.  The metric thresholds below take precedence after calibration.
    proximity_px: int = 80
    danger_proximity_px: int = 28
    overlap_iou: float = 0.01
    warning_distance_m: float = 3.0
    danger_distance_m: float = 1.5
    dynamic_zone_points: int = 28
    consecutive_frames_required: int = 3
    cooldown_seconds: float = 10.0


@dataclass
class IngestConfig:
    max_frame_size_bytes: int = 2_000_000
    target_fps: float = 10.0
    jpeg_quality: int = 80


@dataclass
class PPEConfig:
    model_path: str = "ppe.pt"
    confidence: float = 0.35
    min_person_height_frac: float = 0.30  # bbox height / frame height
    entry_cooldown_sec: float = 4.0       # per-person cooldown between checks


@dataclass
class PostureConfig:
    # Observable coordination/posture anomaly analysis.  This is intentionally
    # not an intoxication classifier; alerts always require human verification.
    enabled: bool = True
    # The behavior model was trained from Pose Landmarker Heavy at 15 Hz.
    # Keep runtime extraction aligned by default; POSTURE_MODEL can switch back
    # to Full/Lite after retraining or an explicit validation on pilot footage.
    model_path: str = "models/pose_landmarker_heavy.task"
    sample_fps: float = 15.0

    # Learned seven-class temporal behavior classifier (TCN).
    behavior_enabled: bool = True
    behavior_model_path: str = "models/behavior/tcn_heavy_ch64/model.pt"
    behavior_device: str = "auto"
    behavior_feature_fps: float = 15.0
    behavior_min_valid_ratio: float = 0.45
    behavior_min_window_coverage: float = 0.70
    behavior_max_sample_gap_seconds: float = 0.35
    behavior_inference_stride_samples: int = 3
    behavior_smoothing_windows: int = 4
    behavior_fall_threshold: float = 0.80
    behavior_lying_threshold: float = 0.85
    behavior_fall_consecutive_windows: int = 1
    behavior_lying_consecutive_windows: int = 3
    behavior_alerts_enabled: bool = True
    max_poses: int = 1
    max_camera_instances: int = 2
    min_person_height_frac: float = 0.18
    min_landmark_visibility: float = 0.50
    min_pose_detection_confidence: float = 0.50
    min_pose_presence_confidence: float = 0.50
    min_tracking_confidence: float = 0.50
    crop_margin: float = 0.24
    crop_center_follow: float = 0.72
    crop_size_follow: float = 0.28
    optical_flow_enabled: bool = True
    optical_flow_win_size: int = 21
    optical_flow_max_level: int = 3
    optical_flow_fb_threshold_px: float = 1.5
    optical_flow_max_jump_frac: float = 0.18

    history_seconds: float = 4.0
    min_history_seconds: float = 2.0
    min_samples: int = 8
    cached_result_ttl_seconds: float = 0.6

    track_ttl_seconds: float = 1.5
    track_min_iou: float = 0.10
    track_max_center_distance: float = 0.55

    observation_score: float = 0.35
    warning_score: float = 0.50
    danger_score: float = 0.82
    consecutive_windows_required: int = 2
    cooldown_seconds: float = 15.0

    # Feature normalization ranges.  Values at *_start begin contributing to
    # the score; values at *_full saturate their component at 1.0.
    torso_sway_start_deg: float = 3.0
    torso_sway_full_deg: float = 12.0
    trajectory_sway_start: float = 0.025
    trajectory_sway_full: float = 0.14
    step_variability_start: float = 0.08
    step_variability_full: float = 0.45
    shoulder_tilt_start_deg: float = 3.0
    shoulder_tilt_full_deg: float = 12.0
    sudden_drop_start: float = 0.35
    sudden_drop_full: float = 1.10
    fall_torso_angle_deg: float = 55.0
    fall_horizontal_fraction: float = 0.45
    fall_drop_component: float = 0.45

    # Detects only a sustained or repeated hand-to-mouth pattern. Human
    # verification is required because radios, drinking and face-touching can
    # produce the same RGB pose-landmark signal.
    hand_to_mouth_enabled: bool = True
    hand_to_mouth_max_ratio: float = 0.65
    hand_to_mouth_fraction: float = 0.55
    hand_to_mouth_window_seconds: float = 2.0
    hand_to_mouth_min_samples: int = 5
    hand_to_mouth_consecutive_windows: int = 2


@dataclass
class WorkerIDConfig:
    # Optional QR-based worker identification.  This deliberately avoids face
    # recognition.  QR payloads must begin with prefix, e.g. worker:W-001.
    enabled: bool = True
    sample_fps: float = 1.5
    cache_ttl_seconds: float = 2.0
    match_padding: float = 0.18
    max_payload_length: int = 96
    require_at_checkpoint: bool = True
    require_on_site: bool = False
    unidentified_frames_required: int = 3
    unidentified_cooldown_seconds: float = 20.0
    # Persistent profile directory keyed by the identifier encoded in QR.
    # This path is inside the Docker ``perimetr_data`` volume by default.
    database_path: str = "data/workers.sqlite3"


@dataclass
class EvidenceConfig:
    """Short evidence clips around confirmed incidents."""
    enabled: bool = True
    clips_dir: str = "data/event_clips"
    pre_seconds: float = 3.0
    post_seconds: float = 3.0
    sample_fps: float = 3.0
    jpeg_quality: int = 70
    max_buffer_frames: int = 90


@dataclass
class Depth3DConfig:
    # Optional monocular metric-depth preview. The small metric V2 model is
    # loaded lazily and uses CUDA FP16 when available.
    enabled: bool = False
    model_id: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    device: str = "cpu"
    local_files_only: bool = False
    min_depth_m: float = 0.20
    max_depth_m: float = 20.0
    visualization_max_depth_m: float = 10.0
    point_stride: int = 5
    export_point_stride: int = 3
    preview_points: int = 4500
    jpeg_quality: int = 82
    target_fps: float = 8.0
    person_distance_enabled: bool = True
    calibration_path: str = "data/depth3d_calibration.json"
    calibration_profiles_dir: str = "data/depth3d_calibrations"


@dataclass
class AppConfig:
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    danger: DangerConfig = field(default_factory=DangerConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    ppe: PPEConfig = field(default_factory=PPEConfig)
    posture: PostureConfig = field(default_factory=PostureConfig)
    worker_id: WorkerIDConfig = field(default_factory=WorkerIDConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    depth3d: Depth3DConfig = field(default_factory=Depth3DConfig)
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
    if v := os.getenv("YOLO_PERSON_CLASS_IDS"):
        cfg.yolo.person_class_ids = [int(x.strip()) for x in v.split(",") if x.strip()]
    if v := os.getenv("YOLO_HAZARD_CLASS_IDS"):
        cfg.yolo.hazard_class_ids = [int(x.strip()) for x in v.split(",") if x.strip()]
    if v := os.getenv("DANGER_PROXIMITY_PX"):
        cfg.danger.proximity_px = int(v)
    if v := os.getenv("DANGER_DANGER_PROXIMITY_PX"):
        cfg.danger.danger_proximity_px = int(v)
    if v := os.getenv("DANGER_WARNING_DISTANCE_M"):
        cfg.danger.warning_distance_m = float(v)
    if v := os.getenv("DANGER_DANGER_DISTANCE_M"):
        cfg.danger.danger_distance_m = float(v)
    if v := os.getenv("DANGER_CONSECUTIVE_FRAMES"):
        cfg.danger.consecutive_frames_required = int(v)
    if v := os.getenv("DANGER_COOLDOWN_SEC"):
        cfg.danger.cooldown_seconds = float(v)
    if v := os.getenv("PPE_MODEL"):
        cfg.ppe.model_path = v
    if v := os.getenv("PPE_CONFIDENCE"):
        cfg.ppe.confidence = float(v)
    if v := os.getenv("PPE_MIN_PERSON_HEIGHT_FRAC"):
        cfg.ppe.min_person_height_frac = float(v)
    if v := os.getenv("PPE_COOLDOWN_SEC"):
        cfg.ppe.entry_cooldown_sec = float(v)
    if v := os.getenv("POSTURE_ENABLED"):
        cfg.posture.enabled = v.strip().lower() in {"1", "true", "yes", "on"}
    if v := os.getenv("POSTURE_MODEL"):
        cfg.posture.model_path = v
    if v := os.getenv("POSTURE_SAMPLE_FPS"):
        cfg.posture.sample_fps = float(v)
    if v := os.getenv("POSTURE_BEHAVIOR_ENABLED"):
        cfg.posture.behavior_enabled = v.strip().lower() in {"1", "true", "yes", "on"}
    if v := os.getenv("POSTURE_BEHAVIOR_MODEL"):
        cfg.posture.behavior_model_path = v
    if v := os.getenv("POSTURE_BEHAVIOR_DEVICE"):
        cfg.posture.behavior_device = v
    if v := os.getenv("POSTURE_BEHAVIOR_FEATURE_FPS"):
        cfg.posture.behavior_feature_fps = max(0.1, float(v))
    if v := os.getenv("POSTURE_BEHAVIOR_MIN_VALID_RATIO"):
        cfg.posture.behavior_min_valid_ratio = min(1.0, max(0.0, float(v)))
    if v := os.getenv("POSTURE_BEHAVIOR_MIN_WINDOW_COVERAGE"):
        cfg.posture.behavior_min_window_coverage = min(1.0, max(0.0, float(v)))
    if v := os.getenv("POSTURE_BEHAVIOR_MAX_SAMPLE_GAP_SEC"):
        cfg.posture.behavior_max_sample_gap_seconds = max(0.01, float(v))
    if v := os.getenv("POSTURE_BEHAVIOR_INFERENCE_STRIDE"):
        cfg.posture.behavior_inference_stride_samples = max(1, int(v))
    if v := os.getenv("POSTURE_BEHAVIOR_SMOOTHING_WINDOWS"):
        cfg.posture.behavior_smoothing_windows = max(1, int(v))
    if v := os.getenv("POSTURE_BEHAVIOR_FALL_THRESHOLD"):
        cfg.posture.behavior_fall_threshold = min(1.0, max(0.0, float(v)))
    if v := os.getenv("POSTURE_BEHAVIOR_LYING_THRESHOLD"):
        cfg.posture.behavior_lying_threshold = min(1.0, max(0.0, float(v)))
    if v := os.getenv("POSTURE_BEHAVIOR_FALL_WINDOWS"):
        cfg.posture.behavior_fall_consecutive_windows = max(1, int(v))
    if v := os.getenv("POSTURE_BEHAVIOR_LYING_WINDOWS"):
        cfg.posture.behavior_lying_consecutive_windows = max(1, int(v))
    if v := os.getenv("POSTURE_BEHAVIOR_ALERTS_ENABLED"):
        cfg.posture.behavior_alerts_enabled = v.strip().lower() in {"1", "true", "yes", "on"}
    if v := os.getenv("POSTURE_MAX_POSES"):
        cfg.posture.max_poses = int(v)
    if v := os.getenv("POSTURE_MIN_PERSON_HEIGHT_FRAC"):
        cfg.posture.min_person_height_frac = float(v)
    if v := os.getenv("POSTURE_CROP_MARGIN"):
        cfg.posture.crop_margin = max(0.0, float(v))
    if v := os.getenv("POSTURE_CROP_CENTER_FOLLOW"):
        cfg.posture.crop_center_follow = min(1.0, max(0.0, float(v)))
    if v := os.getenv("POSTURE_CROP_SIZE_FOLLOW"):
        cfg.posture.crop_size_follow = min(1.0, max(0.0, float(v)))
    if v := os.getenv("POSTURE_OPTICAL_FLOW_ENABLED"):
        cfg.posture.optical_flow_enabled = v.strip().lower() in {"1", "true", "yes", "on"}
    if v := os.getenv("POSTURE_OPTICAL_FLOW_WIN_SIZE"):
        cfg.posture.optical_flow_win_size = max(5, int(v))
    if v := os.getenv("POSTURE_OPTICAL_FLOW_MAX_LEVEL"):
        cfg.posture.optical_flow_max_level = max(0, int(v))
    if v := os.getenv("POSTURE_OPTICAL_FLOW_FB_THRESHOLD_PX"):
        cfg.posture.optical_flow_fb_threshold_px = max(0.1, float(v))
    if v := os.getenv("POSTURE_OPTICAL_FLOW_MAX_JUMP_FRAC"):
        cfg.posture.optical_flow_max_jump_frac = max(0.01, float(v))
    if v := os.getenv("POSTURE_WARNING_SCORE"):
        cfg.posture.warning_score = float(v)
    if v := os.getenv("POSTURE_DANGER_SCORE"):
        cfg.posture.danger_score = float(v)
    if v := os.getenv("POSTURE_CONSECUTIVE_WINDOWS"):
        cfg.posture.consecutive_windows_required = int(v)
    if v := os.getenv("POSTURE_COOLDOWN_SEC"):
        cfg.posture.cooldown_seconds = float(v)
    if v := os.getenv("POSTURE_HAND_TO_MOUTH_ENABLED"):
        cfg.posture.hand_to_mouth_enabled = v.strip().lower() in {"1", "true", "yes", "on"}
    if v := os.getenv("POSTURE_HAND_TO_MOUTH_RATIO"):
        cfg.posture.hand_to_mouth_max_ratio = float(v)
    if v := os.getenv("POSTURE_HAND_TO_MOUTH_FRACTION"):
        cfg.posture.hand_to_mouth_fraction = float(v)
    if v := os.getenv("WORKER_ID_ENABLED"):
        cfg.worker_id.enabled = v.strip().lower() in {"1", "true", "yes", "on"}
    if v := os.getenv("WORKER_ID_SAMPLE_FPS"):
        cfg.worker_id.sample_fps = float(v)
    if v := os.getenv("WORKER_ID_CACHE_TTL_SEC"):
        cfg.worker_id.cache_ttl_seconds = float(v)
    if v := os.getenv("WORKER_ID_REQUIRE_AT_CHECKPOINT"):
        cfg.worker_id.require_at_checkpoint = v.strip().lower() in {"1", "true", "yes", "on"}
    if v := os.getenv("WORKER_ID_REQUIRE_ON_SITE"):
        cfg.worker_id.require_on_site = v.strip().lower() in {"1", "true", "yes", "on"}
    if v := os.getenv("WORKER_ID_UNIDENTIFIED_FRAMES"):
        cfg.worker_id.unidentified_frames_required = int(v)
    if v := os.getenv("WORKER_ID_UNIDENTIFIED_COOLDOWN_SEC"):
        cfg.worker_id.unidentified_cooldown_seconds = float(v)
    if v := os.getenv("WORKER_DATABASE_PATH"):
        cfg.worker_id.database_path = v
    if v := os.getenv("EVIDENCE_ENABLED"):
        cfg.evidence.enabled = v.strip().lower() in {"1", "true", "yes", "on"}
    if v := os.getenv("EVIDENCE_CLIPS_DIR"):
        cfg.evidence.clips_dir = v
    if v := os.getenv("EVIDENCE_PRE_SECONDS"):
        cfg.evidence.pre_seconds = float(v)
    if v := os.getenv("EVIDENCE_POST_SECONDS"):
        cfg.evidence.post_seconds = float(v)
    if v := os.getenv("EVIDENCE_SAMPLE_FPS"):
        cfg.evidence.sample_fps = float(v)
    if v := os.getenv("DEPTH3D_ENABLED"):
        cfg.depth3d.enabled = v.strip().lower() in {"1", "true", "yes", "on"}
    if v := os.getenv("DEPTH3D_MODEL_ID"):
        cfg.depth3d.model_id = v
    if v := os.getenv("DEPTH3D_DEVICE"):
        cfg.depth3d.device = v
    if v := os.getenv("DEPTH3D_LOCAL_FILES_ONLY"):
        cfg.depth3d.local_files_only = v.strip().lower() in {"1", "true", "yes", "on"}
    if v := os.getenv("DEPTH3D_MIN_DEPTH_M"):
        cfg.depth3d.min_depth_m = float(v)
    if v := os.getenv("DEPTH3D_MAX_DEPTH_M"):
        cfg.depth3d.max_depth_m = float(v)
    if v := os.getenv("DEPTH3D_VIS_MAX_DEPTH_M"):
        cfg.depth3d.visualization_max_depth_m = float(v)
    if v := os.getenv("DEPTH3D_POINT_STRIDE"):
        cfg.depth3d.point_stride = max(1, int(v))
    if v := os.getenv("DEPTH3D_TARGET_FPS"):
        cfg.depth3d.target_fps = max(0.1, float(v))
    if v := os.getenv("DEPTH3D_PERSON_DISTANCE_ENABLED"):
        cfg.depth3d.person_distance_enabled = v.strip().lower() in {"1", "true", "yes", "on"}
    if v := os.getenv("DEPTH3D_CALIBRATION_PATH"):
        cfg.depth3d.calibration_path = v
    if v := os.getenv("DEPTH3D_CALIBRATION_PROFILES_DIR"):
        cfg.depth3d.calibration_profiles_dir = v
    if v := os.getenv("SERVER_PORT"):
        cfg.port = int(v)
    return cfg


CONFIG = _from_env()
