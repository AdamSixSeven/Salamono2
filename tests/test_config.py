
import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AppConfig, _from_env


PERFORMANCE_ENV_KEYS = (
    "YOLO_MODEL",
    "YOLO_IMG_SIZE",
    "YOLO_ADAPTIVE_SIZE_ENABLED",
    "YOLO_FALL_RECOVERY_IMG_SIZE",
    "POSTURE_SAMPLE_FPS",
    "POSTURE_MAX_POSES",
    "POSTURE_MIN_PERSON_HEIGHT_FRAC",
    "WORKER_ID_SAMPLE_FPS",
    "WORKER_ID_CACHE_TTL_SEC",
    "WORKER_ID_CROP_PADDING",
    "WORKER_ID_PROFILE_CACHE_TTL_SEC",
    "WORKER_ID_FULL_FRAME_FALLBACK",
    "WORKER_ID_MAX_PENDING_FRAMES",
    "EVIDENCE_SAMPLE_FPS",
)


def test_live_pipeline_defaults_are_latency_balanced():
    cfg = AppConfig()

    assert cfg.yolo.img_size == 512
    assert cfg.yolo.adaptive_size_enabled is True
    assert cfg.yolo.fall_recovery_img_size == 640
    assert cfg.posture.sample_fps == 15.0
    assert cfg.posture.max_poses == 4
    assert cfg.posture.min_person_height_frac == 0.18
    assert cfg.posture.optical_flow_enabled is True
    assert cfg.posture.optical_flow_scale == 0.5
    assert cfg.posture.crop_margin == 0.24
    assert cfg.worker_id.sample_fps == 1.5
    assert cfg.worker_id.cache_ttl_seconds == 4.0
    assert cfg.worker_id.crop_padding == 0.15
    assert cfg.worker_id.profile_cache_ttl_seconds == 60.0
    assert cfg.worker_id.full_frame_fallback is False
    assert cfg.worker_id.max_pending_frames == 1
    assert cfg.evidence.sample_fps == 3.0



def test_custom_scene_detector_defaults_and_hazard_mapping():
    cfg = AppConfig()

    assert cfg.yolo.model_name == "models/perimetr_scene_v3_best.pt"
    assert cfg.yolo.person_class_ids == [0]
    assert cfg.yolo.hazard_class_ids == list(range(1, 10))

    model_path = Path(__file__).resolve().parents[1] / cfg.yolo.model_name
    assert model_path.is_file()


def test_env_example_uses_custom_scene_detector():
    example = (Path(__file__).resolve().parents[1] / ".env.example").read_text(
        encoding="utf-8"
    )
    assert "YOLO_MODEL=models/perimetr_scene_v3_best.pt" in example
    assert "YOLO_PERSON_CLASS_IDS=0" in example
    assert "YOLO_HAZARD_CLASS_IDS=1,2,3,4,5,6,7,8,9" in example

def test_performance_defaults_remain_overridable_and_engine_path_is_supported(
    monkeypatch,
):
    for key in PERFORMANCE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("YOLO_MODEL", "models/yolo11n.engine")
    monkeypatch.setenv("YOLO_IMG_SIZE", "640")
    monkeypatch.setenv("YOLO_ADAPTIVE_SIZE_ENABLED", "false")
    monkeypatch.setenv("YOLO_FALL_RECOVERY_IMG_SIZE", "768")
    monkeypatch.setenv("POSTURE_SAMPLE_FPS", "2.5")
    monkeypatch.setenv("POSTURE_MAX_POSES", "2")
    monkeypatch.setenv("POSTURE_MIN_PERSON_HEIGHT_FRAC", "0.20")
    monkeypatch.setenv("WORKER_ID_SAMPLE_FPS", "1")
    monkeypatch.setenv("WORKER_ID_CACHE_TTL_SEC", "7")
    monkeypatch.setenv("WORKER_ID_CROP_PADDING", "0.25")
    monkeypatch.setenv("WORKER_ID_PROFILE_CACHE_TTL_SEC", "90")
    monkeypatch.setenv("WORKER_ID_FULL_FRAME_FALLBACK", "true")
    monkeypatch.setenv("WORKER_ID_MAX_PENDING_FRAMES", "1")
    monkeypatch.setenv("EVIDENCE_SAMPLE_FPS", "2")

    cfg = _from_env()

    assert cfg.yolo.model_name == "models/yolo11n.engine"
    assert cfg.yolo.img_size == 640
    assert cfg.yolo.adaptive_size_enabled is False
    assert cfg.yolo.fall_recovery_img_size == 768
    assert cfg.posture.sample_fps == 2.5
    assert cfg.posture.max_poses == 2
    assert cfg.posture.min_person_height_frac == 0.20
    assert cfg.worker_id.sample_fps == 1.0
    assert cfg.worker_id.cache_ttl_seconds == 7.0
    assert cfg.worker_id.crop_padding == 0.25
    assert cfg.worker_id.profile_cache_ttl_seconds == 90.0
    assert cfg.worker_id.full_frame_fallback is True
    assert cfg.worker_id.max_pending_frames == 1
    assert cfg.evidence.sample_fps == 2.0


def test_env_example_matches_runtime_performance_defaults():
    example = (

        Path(__file__).resolve().parents[1] / ".env.example"
    ).read_text(encoding="utf-8")

    assert "YOLO_IMG_SIZE=512" in example
    assert "YOLO_ADAPTIVE_SIZE_ENABLED=true" in example
    assert "YOLO_FALL_RECOVERY_IMG_SIZE=640" in example
    assert "POSTURE_SAMPLE_FPS=15" in example
    assert "POSTURE_MAX_POSES=4" in example
    assert "POSTURE_OPTICAL_FLOW_ENABLED=true" in example
    assert "POSTURE_OPTICAL_FLOW_SCALE=0.5" in example
    assert "POSTURE_BEHAVIOR_INFERENCE_STRIDE=3" in example
    assert "POSTURE_CROP_MARGIN=0.24" in example
    assert "POSTURE_MIN_PERSON_HEIGHT_FRAC=0.18" in example
    assert "POSTURE_MIN_PERSON_LONG_SIDE_FRAC=0.10" in example
    assert "POSTURE_MIN_PERSON_AREA_FRAC=0.0025" in example
    assert "WORKER_ID_SAMPLE_FPS=1.5" in example
    assert "WORKER_ID_CACHE_TTL_SEC=4" in example
    assert "WORKER_ID_CROP_PADDING=0.15" in example
    assert "WORKER_ID_PROFILE_CACHE_TTL_SEC=60" in example
    assert "WORKER_ID_FULL_FRAME_FALLBACK=false" in example
    assert "WORKER_ID_MAX_PENDING_FRAMES=1" in example
    assert "WORKER_ID_CONFIRM_SCANS=2" in example
    assert "WORKER_ID_SWITCH_CONFIRM_SCANS=4" in example
    assert "WORKER_ID_DISPLAY_TTL_SEC=5.0" in example
    assert "WORKER_ID_ALERT_MAX_AGE_SEC=2.0" in example
    assert "WORKER_ID_AMBIGUOUS_IOU_THRESHOLD=0.35" in example
    assert "WORKER_ID_ENFORCE_UNIQUE_ACTIVE_ID=true" in example
    assert "WORKER_ID_REACQUIRE_ENABLED=true" in example
    assert "WORKER_ID_REACQUIRE_MAX_GAP_SEC=0.7" in example
    assert "WORKER_ID_SHOW_DEBUG_STATUS=false" in example
    assert "EVIDENCE_SAMPLE_FPS=3" in example
    assert "PERFORMANCE_PROFILING_ENABLED=true" in example


def test_depth3d_defaults_are_indoor_realtime_and_person_aware():
    cfg = AppConfig()
    assert "Metric-Indoor-Small" in cfg.depth3d.model_id
    assert cfg.depth3d.max_depth_m == 20.0
    assert cfg.depth3d.visualization_max_depth_m == 10.0
    assert cfg.depth3d.target_fps == 8.0
    assert cfg.depth3d.person_distance_enabled is True


def test_env_example_activates_indoor_and_comments_out_outdoor():
    example = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
    assert "DEPTH3D_MODEL_ID=depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf" in example
    assert "# DEPTH3D_MODEL_ID=depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf" in example
    assert "DEPTH3D_TARGET_FPS=8" in example
    assert "DEPTH3D_PERSON_DISTANCE_ENABLED=true" in example


def test_demo_video_defaults_and_environment_overrides(monkeypatch):
    overrides = {
        "DEMO_VIDEO_UPLOAD_DIR": "tmp/demo-in",
        "DEMO_VIDEO_OUTPUT_DIR": "tmp/demo-out",
        "DEMO_VIDEO_MAX_SIZE_MB": "321",
        "DEMO_VIDEO_ALLOWED_EXTENSIONS": "mp4,.mov",
        "DEMO_VIDEO_JOB_TTL_SEC": "90",
        "DEMO_VIDEO_MAX_PENDING_JOBS": "3",
        "DEMO_VIDEO_PROCESSING_ENABLED": "false",
        "DEMO_VIDEO_PLAYBACK_MODE": "fast",
        "DEMO_VIDEO_PROBE_TIMEOUT_SEC": "7",
        "DEMO_VIDEO_CONTROL_TIMEOUT_SEC": "4",
    }
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)

    cfg = _from_env().demo_video

    assert cfg.upload_dir == "tmp/demo-in"
    assert cfg.output_dir == "tmp/demo-out"
    assert cfg.max_size_mb == 321
    assert cfg.allowed_extensions == (".mp4", ".mov")
    assert cfg.job_ttl_seconds == 90
    assert cfg.max_pending_jobs == 3
    assert cfg.processing_enabled is False
    assert cfg.playback_mode == "fast"
    assert cfg.probe_timeout_seconds == 7
    assert cfg.control_timeout_seconds == 4


def test_env_example_documents_backend_demo_video_controls():
    example = (Path(__file__).resolve().parents[1] / ".env.example").read_text(
        encoding="utf-8"
    )
    for entry in (
        "DEMO_VIDEO_UPLOAD_DIR=data/tmp/demo_uploads",
        "DEMO_VIDEO_OUTPUT_DIR=data/tmp/demo_outputs",
        "DEMO_VIDEO_MAX_SIZE_MB=2048",
        "DEMO_VIDEO_ALLOWED_EXTENSIONS=.mp4,.avi,.mov,.mkv",
        "DEMO_VIDEO_JOB_TTL_SEC=3600",
        "DEMO_VIDEO_MAX_PENDING_JOBS=2",
        "DEMO_VIDEO_PROCESSING_ENABLED=true",
        "DEMO_VIDEO_PLAYBACK_MODE=realtime",
    ):
        assert entry in example
