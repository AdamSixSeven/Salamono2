#!/usr/bin/env python3
"""Validate and install a compatible pose-event action/safety checkpoint.

Accepted checkpoints:
- original pose-event V2 multitask model,
- V3.2 motion-only train-normalization checkpoint (including seed43 long300).

The runtime path stays stable so existing POSTURE_BEHAVIOR_MODEL configuration
continues to work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = PROJECT_ROOT / "models/behavior/tcn_gru_pose_event_v2"
TARGET_MODEL = TARGET_DIR / "model.pt"

EXPECTED_ACTIONS = [
    "fall_down",
    "lying_down",
    "sit_down",
    "sitting",
    "stand_up",
    "standing",
    "walking",
    "unstable_gait",
    "other",
]
EXPECTED_SAFETY = [
    "normal",
    "unstable_motion",
    "fall_transition",
    "ground_state",
]
V32_MOTION_FORMAT = "perimetr_pose_event_v3_2_motion_ablation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict:
    try:
        import torch
    except Exception as exc:
        raise SystemExit(f"PyTorch is required to validate the checkpoint: {exc}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise SystemExit("Unsupported checkpoint: expected a dictionary")
    return checkpoint


def _validate_common(checkpoint: dict) -> None:
    action_classes = list(checkpoint.get("action_classes", []))
    safety_classes = list(checkpoint.get("safety_classes", []))
    if action_classes != EXPECTED_ACTIONS:
        raise SystemExit(
            "Action classes do not match the selected 9-class model:\n"
            + json.dumps(action_classes, ensure_ascii=False, indent=2)
        )
    if safety_classes != EXPECTED_SAFETY:
        raise SystemExit(
            "Safety classes do not match the selected 4-class model:\n"
            + json.dumps(safety_classes, ensure_ascii=False, indent=2)
        )

    config = dict(checkpoint.get("model_config", {}))
    expected = {
        "input_dim": 387,
        "tcn_channels": 128,
        "gru_hidden": 128,
        "gru_layers": 2,
        "action_classes": 9,
        "safety_classes": 4,
    }
    wrong = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    dropout = float(config.get("dropout", -1.0))
    if abs(dropout - 0.20) > 1e-6:
        wrong["dropout"] = {"expected": 0.20, "actual": dropout}
    if wrong:
        raise SystemExit(
            "Checkpoint architecture does not match the expected 128/128 model:\n"
            + json.dumps(wrong, ensure_ascii=False, indent=2)
        )

    if int(checkpoint.get("frame_count", 0)) != 60:
        raise SystemExit("Checkpoint frame_count must be 60")
    if abs(float(checkpoint.get("fps", 0.0)) - 15.0) > 1e-6:
        raise SystemExit("Checkpoint FPS must be 15")

    if "feature_mean" not in checkpoint or "feature_std" not in checkpoint:
        raise SystemExit("Checkpoint must contain feature_mean and feature_std")


def _validate(checkpoint: dict) -> str:
    if checkpoint.get("format_version") == 2:
        if checkpoint.get("model_type") != "tcn_gru_multitask":
            raise SystemExit("Unsupported V2 checkpoint model_type")
        _validate_common(checkpoint)
        if "state_dict" not in checkpoint:
            raise SystemExit("V2 checkpoint is missing state_dict")
        return "pose-event-v2"

    if checkpoint.get("format") == V32_MOTION_FORMAT:
        _validate_common(checkpoint)
        state = checkpoint.get("motion_state_dict")
        if not isinstance(state, dict):
            raise SystemExit("V3.2 motion checkpoint is missing motion_state_dict")
        missing = [
            key for key in ("motion_branch", "action_head", "safety_head")
            if key not in state
        ]
        if missing:
            raise SystemExit(f"V3.2 motion checkpoint is missing: {missing}")
        return "pose-event-v3.2-motion"

    raise SystemExit(
        "Unsupported checkpoint: expected pose-event V2 or V3.2 motion-only checkpoint"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to a compatible best.pt/best_motion.pt")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an already installed model.pt",
    )
    args = parser.parse_args()

    source = Path(args.checkpoint).expanduser().resolve()
    if not source.is_file():
        print(f"ERROR: checkpoint not found: {source}", file=sys.stderr)
        return 2
    if TARGET_MODEL.exists() and not args.force:
        print(
            f"ERROR: {TARGET_MODEL} already exists; use --force to replace it",
            file=sys.stderr,
        )
        return 3

    checkpoint = _load_checkpoint(source)
    kind = _validate(checkpoint)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, TARGET_MODEL)
    digest = _sha256(TARGET_MODEL)
    (TARGET_DIR / "model.pt.sha256").write_text(
        f"{digest}  model.pt\n", encoding="utf-8"
    )

    for name in (
        "test_report.json",
        "motion_test_report.json",
        "history.json",
        "behavior_history.json",
    ):
        candidate = source.parent / name
        if candidate.is_file():
            shutil.copy2(candidate, TARGET_DIR / name)

    print("Installed:", TARGET_MODEL)
    print("SHA-256:", digest)
    print("Checkpoint:", kind)
    print("Architecture: TCN/GRU 128/128, 2 GRU layers, dropout 0.20")
    print("Runtime heads: 9 action + 4 safety")
    if kind == "pose-event-v3.2-motion":
        metadata = dict(checkpoint.get("metadata") or {})
        print("Normalization:", metadata.get("normalization_source", "checkpoint feature_mean/std"))
        print("Behavior heads: not trained / not used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
