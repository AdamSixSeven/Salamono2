from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
import numpy as np

from .labels import ACTION_CLASSES, SAFETY_CLASSES
from .model import ModelConfig, TCNGRUModel
from .model_v32_motion import V32MotionConfig, V32MotionModel


V32_MOTION_FORMAT = "perimetr_pose_event_v3_2_motion_ablation"


def save_checkpoint(
    path: str | Path,
    model: TCNGRUModel,
    *,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    fps: float,
    frame_count: int,
    action_classes: list[str] | None = None,
    safety_classes: list[str] | None = None,
    action_temperature: float = 1.0,
    safety_temperature: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "model_type": "tcn_gru_multitask",
            "model_config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "feature_mean": np.asarray(feature_mean, dtype=np.float32),
            "feature_std": np.asarray(feature_std, dtype=np.float32),
            "fps": float(fps),
            "frame_count": int(frame_count),
            "action_classes": action_classes or ACTION_CLASSES,
            "safety_classes": safety_classes or SAFETY_CLASSES,
            "action_temperature": float(action_temperature),
            "safety_temperature": float(safety_temperature),
            "metadata": metadata or {},
        },
        path,
    )


def _load_v2(ckpt: dict, device: str | torch.device):
    config = ModelConfig.from_dict(ckpt["model_config"])
    model = TCNGRUModel(config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def _load_v32_motion(ckpt: dict, device: str | torch.device):
    config = V32MotionConfig.from_dict(dict(ckpt["model_config"]))
    model = V32MotionModel(config)

    state = ckpt.get("motion_state_dict")
    if not isinstance(state, dict):
        raise ValueError("V3.2 motion checkpoint is missing motion_state_dict")
    for key in ("motion_branch", "action_head", "safety_head"):
        if key not in state:
            raise ValueError(f"V3.2 motion checkpoint is missing {key}")

    model.motion_branch.load_state_dict(state["motion_branch"], strict=True)
    model.action_head.load_state_dict(state["action_head"], strict=True)
    model.safety_head.load_state_dict(state["safety_head"], strict=True)
    model.to(device).eval()

    # Normalize the metadata contract to the fields expected by PoseEventRuntime.
    normalized = dict(ckpt)
    temperatures = dict(ckpt.get("temperatures") or {})
    normalized.setdefault("action_temperature", float(temperatures.get("action", 1.0)))
    normalized.setdefault("safety_temperature", float(temperatures.get("safety", 1.0)))
    normalized.setdefault("action_classes", list(ACTION_CLASSES))
    normalized.setdefault("safety_classes", list(SAFETY_CLASSES))
    normalized["runtime_model_type"] = "v3_2_motion_only"
    return model, normalized


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError("Unsupported checkpoint: expected a dictionary")

    if ckpt.get("format_version") == 2:
        return _load_v2(ckpt, device)

    if ckpt.get("format") == V32_MOTION_FORMAT:
        return _load_v32_motion(ckpt, device)

    raise ValueError(
        "Unsupported pose-event checkpoint format: expected pose-event v2 or "
        f"{V32_MOTION_FORMAT}"
    )
