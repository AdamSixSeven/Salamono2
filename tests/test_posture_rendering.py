from dataclasses import replace
from unittest.mock import patch

import numpy as np

from backend.detector import Detection
from backend.posture_detector import (
    POSE_LANDMARK_COUNT,
    POSTURE_CONNECTIONS,
    PostureAssessment,
)
from backend.routes.ingest import _annotate_posture, _posture_to_out


def _assessment(pose: np.ndarray) -> PostureAssessment:
    return PostureAssessment(
        track_id=1,
        person=Detection(
            class_id=0,
            class_name="person",
            category="person",
            box=(10, 10, 90, 90),
            confidence=0.95,
        ),
        risk_score=0.0,
        severity="OK",
        status="collecting_history",
        signals=[],
        metrics={},
        frame_timestamp=1.0,
        pose_confidence=0.95,
        history_seconds=0.0,
        landmarks=pose,
    )


def _visible_pose() -> np.ndarray:
    pose = np.zeros((POSE_LANDMARK_COUNT, 4), dtype=np.float32)
    pose[:, 0] = np.linspace(0.1, 0.9, POSE_LANDMARK_COUNT)
    pose[:, 1] = np.linspace(0.9, 0.1, POSE_LANDMARK_COUNT)
    pose[:, 3] = 0.95
    return pose


def test_posture_connections_match_full_mediapipe_pose_topology():
    expected_connections = {
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        (11, 12), (11, 13), (13, 15),
        (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16),
        (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 29), (28, 30), (29, 31), (30, 32),
        (27, 31), (28, 32),
    }

    assert POSE_LANDMARK_COUNT == 33
    assert set(POSTURE_CONNECTIONS) == expected_connections
    assert all(
        0 <= start < POSE_LANDMARK_COUNT and 0 <= end < POSE_LANDMARK_COUNT
        for start, end in POSTURE_CONNECTIONS
    )
    assert (0, 1) in POSTURE_CONNECTIONS       # face
    assert (15, 17) in POSTURE_CONNECTIONS     # left hand
    assert (16, 18) in POSTURE_CONNECTIONS     # right hand
    assert (27, 31) in POSTURE_CONNECTIONS     # left foot
    assert (28, 32) in POSTURE_CONNECTIONS     # right foot


def test_posture_annotation_draws_all_visible_landmarks_and_connections():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with (
        patch("backend.routes.ingest.cv2.line") as draw_line,
        patch("backend.routes.ingest.cv2.circle") as draw_circle,
    ):
        _annotate_posture(frame, [_assessment(_visible_pose())])

    assert draw_line.call_count == len(POSTURE_CONNECTIONS)
    assert draw_circle.call_count == POSE_LANDMARK_COUNT


def test_posture_annotation_skips_hidden_landmark_and_its_connections():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pose = _visible_pose()
    pose[0, 3] = 0.1
    nose_connections = sum(0 in connection for connection in POSTURE_CONNECTIONS)

    with (
        patch("backend.routes.ingest.cv2.line") as draw_line,
        patch("backend.routes.ingest.cv2.circle") as draw_circle,
    ):
        _annotate_posture(frame, [_assessment(pose)])

    assert draw_line.call_count == len(POSTURE_CONNECTIONS) - nose_connections
    assert draw_circle.call_count == POSE_LANDMARK_COUNT - 1


def test_posture_output_exposes_full_pose_for_client_overlay():
    output = _posture_to_out(_assessment(_visible_pose()), "active-1")

    assert len(output.pose_landmarks) == POSE_LANDMARK_COUNT
    assert all(len(landmark) == 4 for landmark in output.pose_landmarks)
    assert output.pose_landmarks[0][3] == 0.95


def test_posture_chip_uses_only_behavior_label_without_track_id():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assessment = replace(
        _assessment(_visible_pose()),
        severity="WARNING",
        status="smoking_detected",
        signals=["smoking_detected"],
        risk_score=0.91,
        behavior_label="standing",
        behavior_confidence=0.93,
        secondary_behavior_probabilities={"phone_call": 0.91, "smoking": 0.88},
    )

    with patch("backend.routes.ingest.cv2.putText") as draw_text:
        _annotate_posture(frame, [assessment])

    label = draw_text.call_args_list[-1].args[1]
    assert label == "standing"
    assert "%" not in label
    assert "TEL" not in label
    assert "SMOKE" not in label
    assert "POSTURE" not in label
    assert "#1" not in label

    with patch("backend.routes.ingest.cv2.putText") as draw_text:
        _annotate_posture(frame, [_assessment(_visible_pose())])
    assert draw_text.call_args_list[-1].args[1] == "analyzing"
