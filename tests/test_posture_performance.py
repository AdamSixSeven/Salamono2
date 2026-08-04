
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.detector import Detection
import backend.posture_detector as posture_module
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
    # Stabilized square crop uses 24% margin and clips to the frame.
    assert estimator.calls[0][0] == (200, 268, 3)
    assert len(result.assessments) == 1

    assessment = result.assessments[0]
    assert assessment.person is largest
    assert assessment.frame_width == 300
    assert assessment.frame_height == 200
    assert assessment.landmarks is not None
    assert np.isclose(assessment.landmarks[0, 0], (32 + 0.5 * 268) / 300)
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


def test_cached_landmarks_are_private_and_translate_without_stretching():
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
    old_center = ((190 + 450) / 2, (80 + 470) / 2)
    new_center = ((210 + 500) / 2, (90 + 470) / 2)
    expected_x = (old_x_px + new_center[0] - old_center[0]) / 640
    expected_y = (old_y_px + new_center[1] - old_center[1]) / 480
    assert np.isclose(new_landmarks[11, 0], expected_x)
    assert np.isclose(new_landmarks[11, 1], expected_y)

    preserved = old_landmarks.copy()
    new_landmarks[11, 0] = 0.0
    assert np.array_equal(old_landmarks, preserved)
    assert cached.assessments[0].confirmed is False
    assert sampled.timings_ms["mediapipe_ms"] > 0.0
    assert sampled.timings_ms["optical_flow_ms"] == 0.0
    assert cached.timings_ms["mediapipe_ms"] == 0.0
    assert cached.timings_ms["optical_flow_ms"] > 0.0
    assert cached.timings_ms["tcn_ms"] == 0.0


def test_analyzer_skips_grayscale_when_no_person_is_posture_eligible(monkeypatch):
    estimator = RecordingEstimator()
    analyzer = PostureAnalyzer(
        estimator,
        _config(
            min_person_height_frac=0.80,
            min_person_long_side_frac=0.80,
            min_person_area_frac=0.50,
        ),
    )
    gray_calls = []
    original = posture_module._scaled_optical_flow_gray

    def recording_gray(frame, scale):
        gray_calls.append((frame.shape, scale))
        return original(frame, scale)

    monkeypatch.setattr(posture_module, "_scaled_optical_flow_gray", recording_gray)
    frame = np.zeros((200, 300, 3), dtype=np.uint8)

    empty = analyzer.process(frame, [], 1.0)
    too_small = analyzer.process(frame, [_person((10, 10, 30, 40))], 2.0)

    assert empty.assessments == []
    assert too_small.assessments == []
    assert estimator.calls == []
    assert gray_calls == []
    assert analyzer._previous_gray is None


def test_analyzer_skips_grayscale_when_cached_pose_no_longer_needs_propagation(
    monkeypatch,
):
    estimator = RecordingEstimator()
    analyzer = PostureAnalyzer(
        estimator,
        _config(sample_fps=5.0, cached_result_ttl_seconds=0.01),
    )
    gray_calls = []
    original = posture_module._scaled_optical_flow_gray

    def recording_gray(frame, scale):
        gray_calls.append((frame.shape, scale))
        return original(frame, scale)

    monkeypatch.setattr(posture_module, "_scaled_optical_flow_gray", recording_gray)
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    person = _person((40, 10, 260, 195))

    sampled = analyzer.process(frame, [person], 1.0)
    expired_cache = analyzer.process(frame, [person], 1.05)

    assert len(sampled.assessments) == 1
    assert expired_cache.inference_ran is False
    assert expired_cache.assessments == []
    # Only the successful MediaPipe sample creates a future flow reference.
    assert len(gray_calls) == 1
    assert expired_cache.timings_ms["optical_flow_ms"] == 0.0


def test_analyzer_builds_scaled_flow_reference_with_default_and_custom_scale():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    person = _person((40, 10, 260, 195))

    default_analyzer = PostureAnalyzer(RecordingEstimator(), _config())
    default_analyzer.process(frame, [person], 1.0)
    assert default_analyzer._previous_gray is not None
    # PostureConfig intentionally has no required field yet: getattr fallback.
    assert default_analyzer._previous_gray.shape == (100, 150)

    custom_config = _config()
    custom_config.optical_flow_scale = 0.25
    custom_analyzer = PostureAnalyzer(RecordingEstimator(), custom_config)
    custom_analyzer.process(frame, [person], 1.0)
    assert custom_analyzer._previous_gray is not None
    assert custom_analyzer._previous_gray.shape == (50, 75)


class BlockingManager:
    def __init__(self):
        self.cfg = _config(max_camera_instances=2)
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[tuple[float, np.ndarray, list[Detection]]] = []
        self.reset_calls: list[str] = []

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

    def reset_camera(self, camera_id: str) -> None:
        self.reset_calls.append(camera_id)


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


def test_posture_worker_reset_waits_and_drops_old_run_snapshot():
    manager = BlockingManager()
    worker = PostureWorker(manager)
    reset_finished = threading.Event()
    reset_result: list[bool] = []

    try:
        assert worker.submit_latest(
            "demo",
            np.zeros((8, 8, 3), dtype=np.uint8),
            [_person((1, 1, 7, 7))],
            1.0,
        )
        assert manager.started.wait(1.0)

        def reset():
            reset_result.append(worker.reset_camera("demo", timeout=1.0))
            reset_finished.set()

        reset_thread = threading.Thread(target=reset)
        reset_thread.start()
        assert not reset_finished.wait(0.05)
        manager.release.set()
        reset_thread.join(1.0)

        assert reset_finished.is_set()
        assert reset_result == [True]
        assert manager.reset_calls == ["demo"]
        assert worker.get_latest_result("demo") is None
    finally:
        manager.release.set()
        worker.close()



def test_optical_flow_tracks_visible_landmarks_between_pose_samples():
    from backend.posture_detector import track_pose_optical_flow

    previous = np.zeros((120, 160), dtype=np.uint8)
    current = np.zeros_like(previous)
    pose = _pose()
    # Put textured blobs at visible landmarks and translate the whole frame.
    for landmark in pose:
        x = int(round(float(landmark[0]) * 160))
        y = int(round(float(landmark[1]) * 120))
        cv = (x, y)
        import cv2
        cv2.circle(previous, cv, 3, 255, -1)
        cv2.circle(current, (x + 4, y + 2), 3, 255, -1)

    fallback = pose.copy()
    tracked, count = track_pose_optical_flow(
        previous,
        current,
        pose,
        fallback,
        frame_width=160,
        frame_height=120,
        bbox=(20, 10, 140, 115),
        min_visibility=0.5,
        fb_threshold_px=2.0,
        max_jump_frac=0.2,
    )

    assert count >= 8
    assert tracked[11, 0] > pose[11, 0]
    assert tracked[11, 1] > pose[11, 1]


def test_optical_flow_maps_full_frame_bbox_and_landmarks_to_scaled_images():
    import cv2
    from backend.posture_detector import track_pose_optical_flow

    full_width, full_height = 160, 120
    flow_width, flow_height = 80, 60
    previous = np.zeros((flow_height, flow_width), dtype=np.uint8)
    current = np.zeros_like(previous)
    pose = _pose()
    for landmark in pose:
        x = int(round(float(landmark[0]) * flow_width))
        y = int(round(float(landmark[1]) * flow_height))
        cv2.circle(previous, (x, y), 2, 255, -1)
        cv2.circle(current, (x + 2, y + 1), 2, 255, -1)

    tracked, count = track_pose_optical_flow(
        previous,
        current,
        pose,
        pose.copy(),
        frame_width=full_width,
        frame_height=full_height,
        bbox=(20, 10, 140, 115),
        min_visibility=0.5,
        fb_threshold_px=2.0,
        max_jump_frac=0.2,
    )

    assert count >= 8
    # A two-pixel shift on the half-width image is still 0.025 normalized.
    assert np.isclose(tracked[11, 0] - pose[11, 0], 2 / flow_width, atol=0.006)
    assert np.isclose(tracked[11, 1] - pose[11, 1], 1 / flow_height, atol=0.006)
    assert np.array_equal(tracked[:, 2:], pose[:, 2:])
