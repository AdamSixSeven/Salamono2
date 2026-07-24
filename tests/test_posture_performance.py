import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.detector import Detection
from backend.posture_detector import (
    PostureAnalyzer,
    PostureProcessResult,
    PostureWorker,
)
from config import PostureConfig


def _pose() -> np.ndarray:
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[:, 0] = 0.5
    pose[:, 1] = 0.5
    pose[:, 3] = 0.95
    pose[11] = [0.43, 0.30, 0.0, 0.95]
    pose[12] = [0.57, 0.30, 0.0, 0.95]
    pose[23] = [0.46, 0.55, 0.0, 0.95]
    pose[24] = [0.54, 0.55, 0.0, 0.95]
    pose[25] = [0.45, 0.72, 0.0, 0.95]
    pose[26] = [0.55, 0.72, 0.0, 0.95]
    pose[27] = [0.44, 0.91, 0.0, 0.95]
    pose[28] = [0.56, 0.91, 0.0, 0.95]
    return pose


def _person(box: tuple[int, int, int, int], confidence: float = 0.95) -> Detection:
    return Detection(
        class_id=0,
        class_name="person",
        category="person",
        box=box,
        confidence=confidence,
    )


class RecordingEstimator:
    def __init__(self):
        self.calls: list[tuple[tuple[int, ...], int, np.ndarray]] = []

    def estimate(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[np.ndarray]:
        self.calls.append((frame_bgr.shape, timestamp_ms, frame_bgr.copy()))
        return [_pose()]

    def close(self) -> None:
        pass


def _config(**updates) -> PostureConfig:
    values = dict(
        enabled=True,
        sample_fps=5.0,
        max_poses=1,
        min_person_height_frac=0.10,
    )
    values.update(updates)
    return PostureConfig(**values)


def test_analyzer_uses_crop_of_largest_qualifying_person_and_maps_pose_back():
    estimator = RecordingEstimator()
    analyzer = PostureAnalyzer(estimator, _config(max_poses=1))
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    small = _person((10, 20, 70, 140))
    largest = _person((100, 20, 280, 190))

    result = analyzer.process(frame, [small, largest], 1.0)

    assert result.inference_ran is True
    assert len(estimator.calls) == 1
    # 18% margin: x=[67, 300], y clipped to [0, 200].
    assert estimator.calls[0][0] == (200, 233, 3)
    assert len(result.assessments) == 1
    assessment = result.assessments[0]
    assert assessment.person is largest
    assert assessment.frame_width == 300
    assert assessment.frame_height == 200
    assert assessment.landmarks is not None
    assert np.isclose(assessment.landmarks[0, 0], (67 + 0.5 * 233) / 300)
    assert np.isclose(assessment.landmarks[0, 1], 0.5)


def test_analyzer_limits_crop_inference_to_configured_largest_people():
    estimator = RecordingEstimator()
    analyzer = PostureAnalyzer(estimator, _config(max_poses=2))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    people = [
        _person((10, 20, 70, 150)),       # area 7,800
        _person((90, 10, 300, 230)),      # area 46,200
        _person((80, 40, 200, 220)),      # area 21,600
    ]

    result = analyzer.process(frame, people, 1.0)

    assert len(estimator.calls) == 2
    assert [assessment.person for assessment in result.assessments] == [
        people[1],
        people[2],
    ]


def test_cached_landmarks_are_private_and_follow_current_person_box():
    estimator = RecordingEstimator()
    analyzer = PostureAnalyzer(estimator, _config(sample_fps=5.0))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    old_person = _person((190, 80, 450, 470))
    new_person = _person((210, 90, 500, 470))

    sampled = analyzer.process(frame, [old_person], 1.0)
    cached = analyzer.process(frame, [new_person], 1.05)

    old_landmarks = sampled.assessments[0].landmarks
    new_landmarks = cached.assessments[0].landmarks
    assert old_landmarks is not None
    assert new_landmarks is not None
    assert not np.shares_memory(old_landmarks, new_landmarks)

    old_x_px = old_landmarks[11, 0] * 640
    old_y_px = old_landmarks[11, 1] * 480
    relative_x = (old_x_px - 190) / (450 - 190)
    relative_y = (old_y_px - 80) / (470 - 80)
    expected_x = (210 + relative_x * (500 - 210)) / 640
    expected_y = (90 + relative_y * (470 - 90)) / 480
    assert np.isclose(new_landmarks[11, 0], expected_x)
    assert np.isclose(new_landmarks[11, 1], expected_y)

    preserved = old_landmarks.copy()
    new_landmarks[11, 0] = 0.0
    assert np.array_equal(old_landmarks, preserved)
    assert cached.assessments[0].confirmed is False


class BlockingManager:
    def __init__(self):
        self.cfg = _config(max_camera_instances=2)
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[tuple[float, np.ndarray, list[Detection]]] = []

    def process(
        self,
        camera_id: str,
        frame: np.ndarray,
        persons: list[Detection],
        timestamp: float,
    ) -> PostureProcessResult:
        if not self.calls:
            self.started.set()
            assert self.release.wait(2.0)
        self.calls.append((timestamp, frame.copy(), persons))
        return PostureProcessResult([], inference_ran=True)

    def close(self) -> None:
        pass


def test_posture_worker_has_one_pending_slot_and_replaces_it_with_latest():
    manager = BlockingManager()
    worker = PostureWorker(manager)
    first_frame = np.full((8, 8, 3), 1, dtype=np.uint8)
    person = _person((1, 1, 7, 7))

    try:
        assert worker.submit_latest("cam-1", first_frame, [person], 1.0)
        assert manager.started.wait(1.0)
        first_frame[:] = 9
        person.confidence = 0.1

        second_frame = np.full((8, 8, 3), 2, dtype=np.uint8)
        third_frame = np.full((8, 8, 3), 3, dtype=np.uint8)
        assert worker.submit_latest("cam-1", second_frame, [person], 2.0)
        assert worker.submit_latest("cam-1", third_frame, [person], 3.0)
        third_frame[:] = 9
        assert worker.pending_count == 1

        manager.release.set()
        latest = worker.wait_for_result(
            "cam-1",
            after_timestamp=1.0,
            timeout=2.0,
        )
        assert latest is not None
        assert latest.frame_timestamp == 3.0
        assert latest.result is not None
        assert latest.error is None
        assert [call[0] for call in manager.calls] == [1.0, 3.0]
        assert np.all(manager.calls[0][1] == 1)
        assert np.all(manager.calls[1][1] == 3)
        assert manager.calls[0][2][0].confidence == 0.95
        assert worker.pending_count == 0
    finally:
        manager.release.set()
        worker.close()
