from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .features import coverage_from_features, extract_frame_features, resample_pose_sequence
from .model_v32_motion import TemporalBranchV32, V32MotionConfig


V32_DUAL_FORMAT = "perimetr_pose_event_v3_2_dual_branch"
V4_DUAL_FORMAT = "perimetr_pose_event_v4_dual_norm_raw"


class V32BehaviorModel(nn.Module):
    """Behavior-only inference wrapper for the V3.2 dual-branch checkpoint."""

    def __init__(self, config: V32MotionConfig):
        super().__init__()
        self.config = config
        self.behavior_branch = TemporalBranchV32(
            input_dim=config.input_dim,
            tcn_channels=config.tcn_channels,
            gru_hidden=config.gru_hidden,
            gru_layers=config.gru_layers,
            shared_dim=config.shared_dim,
            dropout=config.dropout,
        )
        self.smoking_head = nn.Linear(config.shared_dim, 1)
        self.phone_call_head = nn.Linear(config.shared_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        h = self.behavior_branch(x, mask)
        return (
            self.smoking_head(h).squeeze(-1),
            self.phone_call_head(h).squeeze(-1),
        )


@dataclass(frozen=True)
class V32BehaviorPrediction:
    probabilities: dict[str, float]
    valid_ratio: float
    window_seconds: float
    inference_ms: float


def _extract_prefixed(state: dict, prefix: str) -> dict:
    needle = prefix + "."
    out = {
        key[len(needle):]: value
        for key, value in state.items()
        if key.startswith(needle)
    }
    if not out:
        raise ValueError(f"V3.2 checkpoint is missing {prefix}.* weights")
    return out


class V32BehaviorRuntime:
    """Run smoking/phone-call heads from compatible V3.2 or V4 checkpoints."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "auto",
        *,
        min_valid_ratio: float = 0.45,
        max_sample_gap_seconds: float = 0.50,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError("Secondary behavior checkpoint must be a dict")

        checkpoint_format = str(ckpt.get("format") or "")
        if checkpoint_format not in {V32_DUAL_FORMAT, V4_DUAL_FORMAT}:
            raise ValueError(
                "Secondary behavior checkpoint must have format "
                f"{V32_DUAL_FORMAT!r} or {V4_DUAL_FORMAT!r}; "
                f"got {checkpoint_format!r}"
            )

        cfg = V32MotionConfig.from_dict(dict(ckpt["model_config"]))
        model = V32BehaviorModel(cfg)

        if checkpoint_format == V4_DUAL_FORMAT:
            behavior_state = ckpt.get("behavior_state_dict")
            if isinstance(behavior_state, dict) and all(
                key in behavior_state
                for key in ("behavior_branch", "smoking_head", "phone_call_head")
            ):
                model.behavior_branch.load_state_dict(
                    behavior_state["behavior_branch"], strict=True
                )
                model.smoking_head.load_state_dict(
                    behavior_state["smoking_head"], strict=True
                )
                model.phone_call_head.load_state_dict(
                    behavior_state["phone_call_head"], strict=True
                )
            else:
                state = ckpt.get("model_state_dict")
                if not isinstance(state, dict):
                    raise ValueError(
                        "V4 checkpoint is missing behavior_state_dict/model_state_dict"
                    )
                model.behavior_branch.load_state_dict(
                    _extract_prefixed(state, "behavior_branch"), strict=True
                )
                model.smoking_head.load_state_dict(
                    _extract_prefixed(state, "smoking_head"), strict=True
                )
                model.phone_call_head.load_state_dict(
                    _extract_prefixed(state, "phone_call_head"), strict=True
                )

            if "behavior_feature_mean" not in ckpt or "behavior_feature_std" not in ckpt:
                raise ValueError(
                    "V4 checkpoint is missing behavior_feature_mean/std"
                )
            mean = ckpt["behavior_feature_mean"]
            std = ckpt["behavior_feature_std"]
        else:
            state = ckpt.get("model_state_dict")
            if not isinstance(state, dict):
                raise ValueError("V3.2 checkpoint is missing model_state_dict")
            model.behavior_branch.load_state_dict(
                _extract_prefixed(state, "behavior_branch"), strict=True
            )
            model.smoking_head.load_state_dict(
                _extract_prefixed(state, "smoking_head"), strict=True
            )
            model.phone_call_head.load_state_dict(
                _extract_prefixed(state, "phone_call_head"), strict=True
            )
            mean = ckpt["feature_mean"]
            std = ckpt["feature_std"]

        self.model = model.to(self.device).eval()
        self.ckpt = ckpt
        self.checkpoint_format = checkpoint_format

        self.fps = float(ckpt["fps"])
        self.frame_count = int(ckpt["frame_count"])
        self.window_seconds = (self.frame_count - 1) / self.fps
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), 1e-5)
        temps = dict(ckpt.get("temperatures") or {})
        self.smoking_temperature = max(float(temps.get("smoking", 1.0)), 1e-3)
        self.phone_call_temperature = max(float(temps.get("phone_call", 1.0)), 1e-3)
        self.min_valid_ratio = float(min_valid_ratio)
        self.max_sample_gap_seconds = float(max_sample_gap_seconds)

    def predict_history(
        self,
        timestamps: Iterable[float],
        poses: Iterable[np.ndarray],
    ) -> V32BehaviorPrediction | None:
        times = np.asarray(list(timestamps), dtype=np.float64)
        arrays = np.asarray(list(poses), dtype=np.float32)
        if len(times) < 2:
            return None
        if times[-1] - times[0] < self.window_seconds * 0.70:
            return None
        gaps = np.diff(times)
        if len(gaps) and float(np.max(gaps)) > self.max_sample_gap_seconds:
            return None

        sampled, coverage, _ = resample_pose_sequence(
            times,
            arrays,
            self.fps,
            self.frame_count,
            end_time=float(times[-1]),
        )
        features, frame_quality = extract_frame_features(sampled, fps=self.fps)
        valid_ratio = float(np.mean(frame_quality * coverage))
        if valid_ratio < self.min_valid_ratio:
            return None

        covered = coverage > 0.5
        if not np.any(covered):
            return None

        normalized = (features - self.mean) / self.std
        x = torch.from_numpy(normalized[None]).to(self.device)
        mask = torch.from_numpy(covered[None]).to(self.device)

        started = time.perf_counter()
        with torch.inference_mode():
            smoking_logit, phone_logit = self.model(x, mask)
            smoking = torch.sigmoid(smoking_logit / self.smoking_temperature)[0]
            phone = torch.sigmoid(phone_logit / self.phone_call_temperature)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1000.0

        return V32BehaviorPrediction(
            probabilities={
                "smoking": float(smoking.detach().cpu()),
                "phone_call": float(phone.detach().cpu()),
            },
            valid_ratio=valid_ratio,
            window_seconds=self.window_seconds,
            inference_ms=inference_ms,
        )
