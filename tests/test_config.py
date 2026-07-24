import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AppConfig, _from_env


PERFORMANCE_ENV_KEYS = (
    "YOLO_MODEL",
    "YOLO_IMG_SIZE",
    "POSTURE_SAMPLE_FPS",
    "POSTURE_MAX_POSES",
    "POSTURE_MIN_PERSON_HEIGHT_FRAC",
    "WORKER_ID_SAMPLE_FPS",
    "EVIDENCE_SAMPLE_FPS",
)


def test_live_pipeline_defaults_are_latency_balanced():
    cfg = AppConfig()

    assert cfg.yolo.img_size == 512
    assert cfg.posture.sample_fps == 3.0
    assert cfg.posture.max_poses == 1
    assert cfg.posture.min_person_height_frac == 0.25
    assert cfg.worker_id.sample_fps == 1.5
    assert cfg.evidence.sample_fps == 3.0


def test_performance_defaults_remain_overridable_and_engine_path_is_supported(
    monkeypatch,
):
    for key in PERFORMANCE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("YOLO_MODEL", "models/yolo11n.engine")
    monkeypatch.setenv("YOLO_IMG_SIZE", "640")
    monkeypatch.setenv("POSTURE_SAMPLE_FPS", "2.5")
    monkeypatch.setenv("POSTURE_MAX_POSES", "2")
    monkeypatch.setenv("POSTURE_MIN_PERSON_HEIGHT_FRAC", "0.20")
    monkeypatch.setenv("WORKER_ID_SAMPLE_FPS", "1")
    monkeypatch.setenv("EVIDENCE_SAMPLE_FPS", "2")

    cfg = _from_env()

    assert cfg.yolo.model_name == "models/yolo11n.engine"
    assert cfg.yolo.img_size == 640
    assert cfg.posture.sample_fps == 2.5
    assert cfg.posture.max_poses == 2
    assert cfg.posture.min_person_height_frac == 0.20
    assert cfg.worker_id.sample_fps == 1.0
    assert cfg.evidence.sample_fps == 2.0


def test_env_example_matches_runtime_performance_defaults():
    example = (
        Path(__file__).resolve().parents[1] / ".env.example"
    ).read_text(encoding="utf-8")

    assert "YOLO_IMG_SIZE=512" in example
    assert "POSTURE_SAMPLE_FPS=3" in example
    assert "POSTURE_MAX_POSES=1" in example
    assert "POSTURE_MIN_PERSON_HEIGHT_FRAC=0.25" in example
    assert "WORKER_ID_SAMPLE_FPS=1.5" in example
    assert "EVIDENCE_SAMPLE_FPS=3" in example
