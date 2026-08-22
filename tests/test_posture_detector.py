import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.detector import Detection
from backend.posture_detector import (
    PostureAnalyzer,
    PostureAssessment,
    PostureManager,
    _Track,
)
from config import PostureConfig
from pose_event.state_machine import EventDecision


class SequenceEstimator:
    def __init__(self, poses):
        self.poses = list(poses)
        self.index = 0
        self.closed = False

    def estimate(self, frame_bgr, timestamp_ms):
        pose = self.poses[min(self.index, len(self.poses) - 1)]
        self.index += 1
        return [pose.copy()]

    def close(self):
        self.closed = True


def make_pose(sway=0.0, hip_shift=0.0, ankle_sep=0.12, drop=0.0):
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[:, 0] = 0.5
    pose[:, 1] = 0.5
    pose[:, 3] = 0.1

    # Torso: hip remains the reference while shoulders swing sideways.
    pose[11] = [0.45 + sway, 0.32 + drop, 0.0, 0.95]
    pose[12] = [0.55 + sway, 0.32 + drop, 0.0, 0.95]
    pose[23] = [0.47 + hip_shift, 0.55 + drop, 0.0, 0.95]
    pose[24] = [0.53 + hip_shift, 0.55 + drop, 0.0, 0.95]

    pose[25] = [0.46 + hip_shift, 0.70 + drop, 0.0, 0.95]
    pose[26] = [0.54 + hip_shift, 0.70 + drop, 0.0, 0.95]
    pose[27] = [0.50 - ankle_sep / 2, 0.88 + drop, 0.0, 0.95]
    pose[28] = [0.50 + ankle_sep / 2, 0.88 + drop, 0.0, 0.95]
    pose[29] = [0.48 - ankle_sep / 2, 0.90 + drop, 0.0, 0.90]
    pose[30] = [0.52 + ankle_sep / 2, 0.90 + drop, 0.0, 0.90]
    pose[31] = [0.47 - ankle_sep / 2, 0.91 + drop, 0.0, 0.90]
    pose[32] = [0.53 + ankle_sep / 2, 0.91 + drop, 0.0, 0.90]
    return pose


def cfg(**updates):
    base = PostureConfig(
        enabled=True,
        sample_fps=5.0,
        min_person_height_frac=0.10,
        min_history_seconds=1.8,
        min_samples=8,
        consecutive_windows_required=2,
        cooldown_seconds=20.0,
    )
    return replace(base, **updates)


def person():
    return Detection(
        class_id=0,
        class_name="person",
        category="person",
        box=(190, 80, 450, 470),
        confidence=0.95,
    )


def run_sequence(poses, config=None):
    analyzer = PostureAnalyzer(SequenceEstimator(poses), config or cfg())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    outputs = []
    for i in range(len(poses)):
        result = analyzer.process(frame, [person()], i * 0.2)
        outputs.extend(result.assessments)
    return outputs


def test_stable_pose_remains_normal():
    outputs = run_sequence([make_pose() for _ in range(18)])
    mature = [o for o in outputs if o.status != "collecting_history"]
    assert mature
    assert mature[-1].risk_score < 0.2
    assert mature[-1].status == "normal"
    assert mature[-1].signals == []
    assert not any(o.confirmed for o in mature)


def test_repeated_sway_creates_human_verification_alert():
    poses = []
    import math
    for i in range(30):
        sway_wave = math.sin(i * 0.7)
        step_wave = abs(math.sin(i * 1.15))
        poses.append(make_pose(
            # The estimator now receives a person crop.  These crop-relative
            # amplitudes describe the same pronounced sway/trajectory pattern
            # that the earlier full-frame fixture represented with 0.10/0.06.
            sway=0.18 * sway_wave,
            hip_shift=0.10 * sway_wave,
            ankle_sep=0.08 + 0.22 * step_wave,
        ))

    outputs = run_sequence(poses)
    mature = [o for o in outputs if o.status != "collecting_history"]
    assert mature
    assert mature[-1].risk_score >= cfg().warning_score
    assert "repeated_body_sway" in mature[-1].signals
    assert "unstable_trajectory" in mature[-1].signals
    assert any(o.confirmed for o in mature)


def test_sampling_returns_cached_score_without_duplicate_confirmation():
    estimator = SequenceEstimator([make_pose()] * 4)
    analyzer = PostureAnalyzer(estimator, cfg())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    first = analyzer.process(frame, [person()], 0.0)
    cached = analyzer.process(frame, [person()], 0.05)

    assert first.inference_ran is True
    assert cached.inference_ran is False
    assert estimator.index == 1
    assert len(cached.assessments) == 1
    assert cached.assessments[0].landmarks is not None
    assert np.array_equal(
        cached.assessments[0].landmarks,
        first.assessments[0].landmarks,
    )
    assert not np.shares_memory(
        cached.assessments[0].landmarks,
        first.assessments[0].landmarks,
    )
    assert cached.assessments[0].confirmed is False


def test_partially_occluded_pose_is_displayed_but_not_analyzed():
    low_visibility_pose = make_pose()
    low_visibility_pose[:, 3] = 0.35
    analyzer = PostureAnalyzer(
        SequenceEstimator([low_visibility_pose]),
        cfg(
            min_landmark_visibility=0.30,
            min_analysis_landmark_visibility=0.50,
        ),
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = analyzer.process(frame, [person()], 0.0)

    assert len(result.assessments) == 1
    assessment = result.assessments[0]
    assert assessment.status == "insufficient_pose"
    assert assessment.landmarks is not None
    assert assessment.signals == []
    assert assessment.confirmed is False
    assert assessment.metrics["analysis_landmarks_valid"] == 0.0
    assert not analyzer._tracks[assessment.track_id].history


def test_cached_learned_confirmation_is_emitted_only_once():
    analyzer = PostureAnalyzer(
        SequenceEstimator([make_pose()]),
        cfg(learned_events_enabled=True, heuristic_alerts_enabled=False),
    )
    track = _Track(track_id=7, last_box=person().box, last_seen=10.0)
    track.behavior_prediction_version = 3
    track.behavior_last_prediction_at = 10.0
    track.learned_decision = EventDecision(
        event_type="fall_detected",
        severity="DANGER",
        status="confirmed_direct_fall",
        confirmed=True,
        score=0.99,
        reason="test confirmation",
    )
    assessment = PostureAssessment(
        track_id=7,
        person=person(),
        risk_score=0.0,
        severity="OK",
        status="normal",
        signals=[],
        metrics={},
        frame_timestamp=10.0,
        pose_confidence=0.95,
        history_seconds=2.0,
    )

    first = analyzer._merge_behavior_assessment(track, assessment, 10.0)
    cached = analyzer._merge_behavior_assessment(track, assessment, 10.1)

    assert first.confirmed is True
    assert cached.confirmed is False
    assert cached.learned_event_type == "fall_detected"
    assert cached.severity == "DANGER"

    # A genuinely new classifier decision gets its own single emission.
    track.behavior_prediction_version += 1
    new_prediction = analyzer._merge_behavior_assessment(track, assessment, 10.2)
    replay = analyzer._merge_behavior_assessment(track, assessment, 10.3)
    assert new_prediction.confirmed is True
    assert replay.confirmed is False


def test_phone_or_smoking_probability_creates_one_smoking_alert_at_threshold():
    analyzer = PostureAnalyzer(
        SequenceEstimator([make_pose()]),
        cfg(learned_events_enabled=False, heuristic_alerts_enabled=False),
    )
    assessment = PostureAssessment(
        track_id=7,
        person=person(),
        risk_score=0.0,
        severity="OK",
        status="normal",
        signals=[],
        metrics={},
        frame_timestamp=10.0,
        pose_confidence=0.95,
        history_seconds=2.0,
    )

    for probabilities in (
        {"phone_call": 0.85, "smoking": 0.1},
        {"phone_call": 0.90, "smoking": 0.91},
    ):
        track = _Track(track_id=7, last_box=person().box, last_seen=10.0)
        track.behavior_prediction_version = 1
        track.behavior_last_prediction_at = 10.0
        track.secondary_behavior_probabilities = probabilities

        result = analyzer._merge_behavior_assessment(track, assessment, 10.0)
        replay = analyzer._merge_behavior_assessment(track, assessment, 10.1)

        assert result.signals == ["smoking_detected"]
        assert result.severity == "WARNING"
        assert result.status == "smoking_detected"
        assert result.confirmed is True
        assert replay.confirmed is False

    below_threshold = _Track(
        track_id=8,
        last_box=person().box,
        last_seen=10.0,
    )
    below_threshold.behavior_prediction_version = 1
    below_threshold.behavior_last_prediction_at = 10.0
    below_threshold.secondary_behavior_probabilities = {
        "phone_call": 0.849,
        "smoking": 0.849,
    }
    result = analyzer._merge_behavior_assessment(
        below_threshold,
        replace(assessment, track_id=8),
        10.0,
    )
    assert "smoking_detected" not in result.signals
    assert result.confirmed is False


def test_manager_accepts_injected_estimator_without_mediapipe_model():
    estimator = SequenceEstimator([make_pose()])
    manager = PostureManager(
        cfg(),
        estimator_factory=lambda _config: estimator,
    )
    assert manager.available is True

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = manager.process("cam_test", frame, [person()], 0.0)
    assert len(result.assessments) == 1
    manager.close()
    assert estimator.closed is True


def make_fallen_pose():
    pose = make_pose()
    # Horizontal torso after a rapid downward hip displacement.
    pose[11] = [0.60, 0.75, 0.0, 0.95]
    pose[12] = [0.70, 0.75, 0.0, 0.95]
    pose[23] = [0.47, 0.75, 0.0, 0.95]
    pose[24] = [0.53, 0.75, 0.0, 0.95]
    pose[25] = [0.45, 0.82, 0.0, 0.95]
    pose[26] = [0.55, 0.82, 0.0, 0.95]
    pose[27] = [0.42, 0.88, 0.0, 0.95]
    pose[28] = [0.58, 0.88, 0.0, 0.95]
    return pose


def test_sudden_drop_followed_by_horizontal_torso_flags_possible_fall():
    poses = [make_pose() for _ in range(12)] + [make_fallen_pose() for _ in range(10)]
    outputs = run_sequence(poses)
    mature = [output for output in outputs if output.status != "collecting_history"]
    assert mature
    assert any("possible_fall" in output.signals for output in mature)
    fall_outputs = [output for output in mature if "possible_fall" in output.signals]
    assert fall_outputs[-1].severity == "DANGER"
    assert any(output.confirmed for output in fall_outputs)


def make_hand_to_mouth_pose(near=True):
    pose = make_pose()
    pose[9] = [0.49, 0.24, 0.0, 0.95]
    pose[10] = [0.51, 0.24, 0.0, 0.95]
    if near:
        pose[15] = [0.50, 0.27, 0.0, 0.95]
    else:
        pose[15] = [0.35, 0.65, 0.0, 0.95]
    pose[16] = [0.65, 0.65, 0.0, 0.95]
    return pose


def test_repeated_hand_to_mouth_pattern_creates_verification_candidate():
    poses = [make_hand_to_mouth_pose(False) for _ in range(8)]
    poses += [make_hand_to_mouth_pose(True) for _ in range(14)]
    outputs = run_sequence(poses)
    candidates = [o for o in outputs if "hand_to_mouth_pattern" in o.signals]
    assert candidates
    assert candidates[-1].severity == "WARNING"
    assert candidates[-1].status == "verification_required"
    assert candidates[-1].metrics["hand_to_mouth_fraction"] >= 0.55
    assert any(o.confirmed for o in candidates)


class TrackAwareEstimator:
    def __init__(self):
        self.track_ids = []
        self.closed = False

    def estimate_for_track(self, frame_bgr, timestamp_ms, track_id):
        self.track_ids.append(track_id)
        return [make_pose()]

    def estimate(self, frame_bgr, timestamp_ms):
        raise AssertionError("multi-track posture path should be used")

    def close(self):
        self.closed = True


def test_posture_analyzes_four_independent_person_boxes():
    estimator = TrackAwareEstimator()
    config = cfg(max_poses=4, sample_fps=15.0)
    analyzer = PostureAnalyzer(estimator, config)
    frame = np.zeros((1000, 3200, 3), dtype=np.uint8)
    people = [
        Detection(0, "person", "person", (400, 200, 620, 800), 0.95),
        Detection(0, "person", "person", (1050, 200, 1270, 800), 0.95),
        Detection(0, "person", "person", (1700, 200, 1920, 800), 0.95),
        Detection(0, "person", "person", (2350, 200, 2570, 800), 0.95),
        # Fifth person is deliberately smaller and must be skipped by max_poses.
        Detection(0, "person", "person", (2850, 500, 2950, 800), 0.95),
    ]

    result = analyzer.process(frame, people, 0.0)

    assert result.inference_ran is True
    assert len(result.assessments) == 4
    assert len(set(estimator.track_ids)) == 4
    assert len(estimator.track_ids) == 4
