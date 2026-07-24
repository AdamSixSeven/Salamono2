import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.detector import Detection
from backend.posture_detector import PostureAnalyzer, PostureManager
from config import PostureConfig


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
