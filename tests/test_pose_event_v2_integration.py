from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from backend.learned_event_controller import LearnedEventController


def prediction(
    *,
    normal=0.0,
    unstable=0.0,
    fall=0.0,
    ground=0.0,
    action_fall=0.0,
    torso=1.0,
    lower=1.0,
):
    return SimpleNamespace(
        probabilities={"fall_down": action_fall},
        safety_probabilities={
            "normal": normal,
            "unstable_motion": unstable,
            "fall_transition": fall,
            "ground_state": ground,
        },
        valid_ratio=1.0,
        torso_quality=torso,
        lower_body_quality=lower,
    )


def test_fall_transition_followed_by_ground_confirms_event():
    controller = LearnedEventController(
        fall_threshold=0.60,
        ground_threshold=0.60,
        ground_confirm_seconds=0.40,
        fall_followup_seconds=3.0,
        cooldown_seconds=10.0,
    )

    first = controller.update(
        7,
        1.0,
        prediction(fall=0.90, ground=0.05, normal=0.05),
    )
    assert first.event_type == "fall_suspected"
    assert first.confirmed is False

    controller.update(7, 1.2, prediction(ground=0.90, fall=0.05, normal=0.05))
    controller.update(7, 1.4, prediction(ground=0.90, fall=0.05, normal=0.05))
    final = controller.update(
        7,
        1.6,
        prediction(ground=0.90, fall=0.05, normal=0.05),
    )
    assert final.event_type == "fall_detected"
    assert final.severity == "DANGER"
    assert final.confirmed is True


def test_persistent_dual_head_fall_confirms_without_ground_state():
    controller = LearnedEventController(
        fall_threshold=0.60,
        direct_fall_threshold=0.85,
        direct_fall_action_threshold=0.70,
        direct_fall_confirm_seconds=0.60,
        ground_threshold=0.70,
        cooldown_seconds=10.0,
    )

    first = controller.update(
        11,
        1.0,
        prediction(fall=0.95, action_fall=0.82, normal=0.03),
    )
    assert first.event_type == "fall_suspected"
    assert first.confirmed is False

    controller.update(
        11,
        1.3,
        prediction(fall=0.96, action_fall=0.83, normal=0.02),
    )
    final = controller.update(
        11,
        1.7,
        prediction(fall=0.97, action_fall=0.84, normal=0.01),
    )
    assert final.event_type == "fall_detected"
    assert final.status == "confirmed_direct_fall"
    assert final.confirmed is True


def test_persistent_ground_state_is_confirmed_without_claiming_fall():
    controller = LearnedEventController(
        ground_threshold=0.70,
        ground_confirm_seconds=0.50,
        cooldown_seconds=10.0,
    )
    controller.update(4, 1.0, prediction(ground=0.90, normal=0.10))
    controller.update(4, 1.3, prediction(ground=0.92, normal=0.08))
    final = controller.update(4, 1.6, prediction(ground=0.94, normal=0.06))

    assert final.event_type == "person_on_ground"
    assert final.severity == "DANGER"
    assert final.confirmed is True


def test_unstable_event_is_suppressed_when_legs_are_not_visible():
    controller = LearnedEventController(
        unstable_threshold=0.60,
        unstable_confirm_seconds=0.20,
        min_lower_body_quality_for_unstable=0.35,
    )
    result = controller.update(
        "worker-1",
        2.0,
        prediction(unstable=0.95, normal=0.05, lower=0.10),
    )
    assert result.event_type is None
    assert result.status == "insufficient_lower_body_pose"
    assert result.confirmed is False


@pytest.mark.parametrize("pose_columns", [4, 5])
def test_behavior_classifier_loads_pose_event_v2_checkpoint(tmp_path, pose_columns):
    torch = pytest.importorskip("torch")

    from backend.behavior_classifier import BehaviorClassifier
    from pose_event.checkpoint import save_checkpoint
    from pose_event.labels import ACTION_CLASSES, SAFETY_CLASSES
    from pose_event.model import ModelConfig, TCNGRUModel

    model = TCNGRUModel(
        ModelConfig(
            input_dim=387,
            tcn_channels=16,
            gru_hidden=16,
            gru_layers=1,
            dropout=0.0,
            action_classes=len(ACTION_CLASSES),
            safety_classes=len(SAFETY_CLASSES),
        )
    )
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        model,
        feature_mean=np.zeros(387, dtype=np.float32),
        feature_std=np.ones(387, dtype=np.float32),
        fps=15.0,
        frame_count=60,
        action_classes=ACTION_CLASSES,
        safety_classes=SAFETY_CLASSES,
    )

    classifier = BehaviorClassifier(checkpoint, device="cpu")
    pose = np.zeros((33, pose_columns), dtype=np.float32)
    pose[:, 3] = 1.0
    if pose_columns == 5:
        pose[:, 4] = 1.0
    pose[11, :3] = [0.45, 0.30, 0.0]
    pose[12, :3] = [0.55, 0.30, 0.0]
    pose[23, :3] = [0.47, 0.55, 0.0]
    pose[24, :3] = [0.53, 0.55, 0.0]
    pose[25, :3] = [0.47, 0.70, 0.0]
    pose[26, :3] = [0.53, 0.70, 0.0]
    pose[27, :3] = [0.46, 0.88, 0.0]
    pose[28, :3] = [0.54, 0.88, 0.0]

    history = [(index / 15.0, pose.copy()) for index in range(60)]
    result = classifier.predict_history(history)

    assert result is not None
    assert result.label in ACTION_CLASSES
    assert result.safety_label in SAFETY_CLASSES
    assert set(result.probabilities) == set(ACTION_CLASSES)
    assert set(result.safety_probabilities) == set(SAFETY_CLASSES)
    assert torch.isfinite(torch.tensor(result.confidence))
