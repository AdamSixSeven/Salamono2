from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np

from backend.detector import Detection
from backend.latest_detection_store import LatestDetectionStore
from backend.routes.depth3d import infer


class _LatestFrameStore:
    def __init__(self, frame: np.ndarray):
        self.item = SimpleNamespace(
            frame=frame,
            width=frame.shape[1],
            height=frame.shape[0],
            timestamp=time.time(),
            received_at=time.time(),
        )

    def get(self, camera_id: str, max_age_seconds: float | None = None):
        return self.item


class _CalibrationStore:
    def get(self, camera_id: str):
        return None


class _ResultStore:
    def __init__(self):
        self.result = None

    def put(self, result):
        self.result = result


class _Estimator:
    model_id = "test-indoor-model"

    def infer(self, frame: np.ndarray) -> np.ndarray:
        return np.full(frame.shape[:2], 3.0, dtype=np.float32)

    def status(self):
        return {"device": "cuda:0", "precision": "fp16"}


class _Detector:
    def detect(self, frame: np.ndarray):
        return [
            Detection(
                class_id=0,
                class_name="person",
                category="person",
                box=(20, 10, 80, 90),
                confidence=0.91,
            )
        ]


def test_infer_returns_person_distance_and_jpeg_depth():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    state = SimpleNamespace(
        latest_frame_store=_LatestFrameStore(frame),
        depth3d_calibration_store=_CalibrationStore(),
        depth3d_result_store=_ResultStore(),
        depth3d_estimator=_Estimator(),
        detector=_Detector(),
        analysis_lock=threading.Lock(),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    payload = asyncio.run(infer("cam-test", request))

    assert payload["depth_image_mime"] == "image/jpeg"
    assert payload["depth_jpeg_b64"]
    assert payload["backend_total_ms"] >= payload["processing_ms"]
    assert len(payload["person_distances"]) == 1
    person = payload["person_distances"][0]
    assert person["depth_z_m"] == 3.0
    assert person["ray_distance_m"] >= 3.0
    assert person["confidence"] == 0.91


def test_infer_reuses_exact_frame_yolo_result_instead_of_running_detector_twice():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    latest = _LatestFrameStore(frame)
    detections = LatestDetectionStore()
    detections.put(
        "cam-cache",
        latest.item.timestamp,
        100,
        100,
        [_Detector().detect(frame)[0]],
    )

    class _UnexpectedDetector:
        def detect(self, _frame):
            raise AssertionError("exact-frame YOLO cache should be reused")

    state = SimpleNamespace(
        latest_frame_store=latest,
        latest_detection_store=detections,
        depth3d_calibration_store=_CalibrationStore(),
        depth3d_result_store=_ResultStore(),
        depth3d_estimator=_Estimator(),
        detector=_UnexpectedDetector(),
        analysis_lock=threading.Lock(),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    payload = asyncio.run(infer("cam-cache", request))

    assert len(payload["person_distances"]) == 1
    assert payload["person_distances"][0]["confidence"] == 0.91
