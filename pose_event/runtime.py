from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .features import (
    coverage_from_features,
    extract_frame_features,
    resample_pose_sequence,
)


@dataclass
class PoseEventPrediction:
    action_label: str
    action_confidence: float
    action_probabilities: dict[str, float]
    safety_label: str
    safety_confidence: float
    safety_probabilities: dict[str, float]
    valid_ratio: float
    torso_quality: float
    upper_body_quality: float
    lower_body_quality: float
    visible_ratio: float
    window_seconds: float
    inference_ms: float


@dataclass
class _TrackBuffer:
    timestamps: deque[float] = field(default_factory=deque)
    poses: deque[np.ndarray] = field(default_factory=deque)


class PoseEventRuntime:
    """Per-track runtime around the occlusion-aware TCN+GRU checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "auto",
        *,
        min_valid_ratio: float = 0.45,
        max_sample_gap_seconds: float = 0.50,
        history_margin_seconds: float = 1.0,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, self.ckpt = load_checkpoint(checkpoint, self.device)
        self.fps = float(self.ckpt["fps"])
        self.frame_count = int(self.ckpt["frame_count"])
        self.window_seconds = (self.frame_count - 1) / self.fps
        self.action_classes = list(self.ckpt["action_classes"])
        self.safety_classes = list(self.ckpt["safety_classes"])
        self.mean = np.asarray(self.ckpt["feature_mean"], dtype=np.float32)
        self.std = np.maximum(np.asarray(self.ckpt["feature_std"], dtype=np.float32), 1e-5)
        self.action_temperature = max(float(self.ckpt.get("action_temperature", 1.0)), 1e-3)
        self.safety_temperature = max(float(self.ckpt.get("safety_temperature", 1.0)), 1e-3)
        self.min_valid_ratio = float(min_valid_ratio)
        self.max_sample_gap_seconds = float(max_sample_gap_seconds)
        self.max_history_seconds = self.window_seconds + float(history_margin_seconds)
        self._tracks: dict[int | str, _TrackBuffer] = {}

    def reset(self, track_id: int | str | None = None) -> None:
        if track_id is None:
            self._tracks.clear()
        else:
            self._tracks.pop(track_id, None)

    def add_sample(
        self,
        track_id: int | str,
        timestamp: float,
        pose: np.ndarray,
    ) -> PoseEventPrediction | None:
        arr = np.asarray(pose, dtype=np.float32)
        if arr.shape not in ((33, 4), (33, 5)):
            raise ValueError(f"Expected pose 33×4/5, got {arr.shape}")
        state = self._tracks.setdefault(track_id, _TrackBuffer())
        timestamp = float(timestamp)
        if state.timestamps and timestamp <= state.timestamps[-1]:
            timestamp = state.timestamps[-1] + 1e-4
        state.timestamps.append(timestamp)
        state.poses.append(arr.copy())
        cutoff = timestamp - self.max_history_seconds
        while state.timestamps and state.timestamps[0] < cutoff:
            state.timestamps.popleft()
            state.poses.popleft()
        return self.predict_history(state.timestamps, state.poses)

    def predict_history(
        self,
        timestamps: Iterable[float],
        poses: Iterable[np.ndarray],
    ) -> PoseEventPrediction | None:
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
            times, arrays, self.fps, self.frame_count, end_time=float(times[-1])
        )
        features, frame_quality = extract_frame_features(sampled, fps=self.fps)
        valid_ratio = float(np.mean(frame_quality * coverage))
        if valid_ratio < self.min_valid_ratio:
            return None

        body = coverage_from_features(features)
        covered = coverage > 0.5
        if not np.any(covered):
            return None

        def mean_covered(values):
            return float(np.mean(np.asarray(values)[covered]))

        torso_quality = mean_covered(body["torso_coverage"])
        upper_body_quality = mean_covered(
            0.5 * (body["left_arm_coverage"] + body["right_arm_coverage"])
        )
        lower_body_quality = mean_covered(
            0.5 * (body["left_leg_coverage"] + body["right_leg_coverage"])
        )
        visible_ratio = mean_covered(body["visible_ratio"])

        normalized = (features - self.mean) / self.std
        x = torch.from_numpy(normalized[None]).to(self.device)
        mask = torch.from_numpy((coverage > 0.5)[None]).to(self.device)

        started = time.perf_counter()
        with torch.inference_mode():
            action_logits, safety_logits = self.model(x, mask)
            action_probs = torch.softmax(action_logits / self.action_temperature, dim=1)[0]
            safety_probs = torch.softmax(safety_logits / self.safety_temperature, dim=1)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1000.0

        action_np = action_probs.detach().cpu().numpy()
        safety_np = safety_probs.detach().cpu().numpy()
        ai = int(np.argmax(action_np))
        si = int(np.argmax(safety_np))
        return PoseEventPrediction(
            action_label=self.action_classes[ai],
            action_confidence=float(action_np[ai]),
            action_probabilities={
                name: float(action_np[i]) for i, name in enumerate(self.action_classes)
            },
            safety_label=self.safety_classes[si],
            safety_confidence=float(safety_np[si]),
            safety_probabilities={
                name: float(safety_np[i]) for i, name in enumerate(self.safety_classes)
            },
            valid_ratio=valid_ratio,
            torso_quality=torso_quality,
            upper_body_quality=upper_body_quality,
            lower_body_quality=lower_body_quality,
            visible_ratio=visible_ratio,
            window_seconds=self.window_seconds,
            inference_ms=inference_ms,
        )
