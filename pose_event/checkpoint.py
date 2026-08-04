from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
import numpy as np

from .labels import ACTION_CLASSES, SAFETY_CLASSES
from .model import ModelConfig, TCNGRUModel


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


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    if ckpt.get("format_version") != 2:
        raise ValueError("Unsupported checkpoint format; expected pose-event v2")
    config = ModelConfig.from_dict(ckpt["model_config"])
    model = TCNGRUModel(config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
