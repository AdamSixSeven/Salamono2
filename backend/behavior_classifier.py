"""Runtime TCN classifier for timestamped MediaPipe pose sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable, Sequence

import numpy as np

from pose_behavior.features import extract_frame_features
from pose_behavior.tcn import PoseTCN


@dataclass(frozen=True)
class BehaviorPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    valid_ratio: float
    window_seconds: float
    inference_ms: float


class BehaviorClassifier:
    """Load a trained PoseTCN checkpoint and classify timestamped poses."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        feature_fps: float = 15.0,
        min_valid_ratio: float = 0.45,
        min_window_coverage: float = 0.70,
        max_sample_gap_seconds: float = 0.35,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Behavior model not found: {path}")
        if feature_fps <= 0:
            raise ValueError("feature_fps must be positive")

        try:
            import torch
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(f"PyTorch is unavailable: {exc}") from exc

        self._torch = torch
        self.model_path = str(path)
        self.feature_fps = float(feature_fps)
        self.min_valid_ratio = float(min_valid_ratio)
        self.min_window_coverage = float(min_window_coverage)
        self.max_sample_gap_seconds = float(max_sample_gap_seconds)

        requested = (device or "auto").strip().lower()
        if requested in {"auto", ""}:
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # older PyTorch releases
            checkpoint = torch.load(path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError("Unsupported behavior checkpoint format")

        required = {
            "state_dict",
            "labels",
            "feature_names",
            "mean",
            "std",
            "input_features",
            "window_frames",
            "channels",
            "dropout",
        }
        missing = sorted(required - set(checkpoint))
        if missing:
            raise ValueError(f"Behavior checkpoint is missing keys: {missing}")

        self.labels = [str(label) for label in checkpoint["labels"]]
        self.feature_names = [str(name) for name in checkpoint["feature_names"]]
        self.input_features = int(checkpoint["input_features"])
        self.window_frames = int(checkpoint["window_frames"])
        self.channels = int(checkpoint["channels"])
        self.dropout = float(checkpoint["dropout"])
        if self.window_frames < 2:
            raise ValueError("Behavior checkpoint window must contain >= 2 frames")
        if self.input_features != len(self.feature_names):
            raise ValueError("Checkpoint feature_names width does not match input_features")

        mean = np.asarray(checkpoint["mean"], dtype=np.float32).reshape(1, 1, -1)
        std = np.asarray(checkpoint["std"], dtype=np.float32).reshape(1, 1, -1)
        if mean.shape[-1] != self.input_features or std.shape[-1] != self.input_features:
            raise ValueError("Checkpoint normalization width does not match input_features")
        self._mean = mean
        self._std = np.where(np.abs(std) > 1e-6, std, 1.0).astype(np.float32)

        self.model = PoseTCN(
            input_features=self.input_features,
            num_classes=len(self.labels),
            channels=self.channels,
            dropout=self.dropout,
        )
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

    @property
    def target_window_seconds(self) -> float:
        return (self.window_frames - 1) / self.feature_fps

    def _resample_history(
        self,
        history: Sequence[tuple[float, np.ndarray]],
    ) -> tuple[np.ndarray, float] | None:
        cleaned: list[tuple[float, np.ndarray]] = []
        for timestamp, pose in history:
            timestamp = float(timestamp)
            arr = np.asarray(pose, dtype=np.float32)
            if not np.isfinite(timestamp) or arr.ndim != 2 or arr.shape[0] != 33 or arr.shape[1] < 4:
                continue
            cleaned.append((timestamp, arr[:, :4].copy()))
        if len(cleaned) < 2:
            return None

        cleaned.sort(key=lambda item: item[0])
        deduplicated: list[tuple[float, np.ndarray]] = []
        for item in cleaned:
            if deduplicated and abs(item[0] - deduplicated[-1][0]) < 1e-9:
                deduplicated[-1] = item
            else:
                deduplicated.append(item)
        if len(deduplicated) < 2:
            return None

        end_time = deduplicated[-1][0]
        start_time = end_time - self.target_window_seconds
        selected = [
            item for item in deduplicated
            if item[0] >= start_time - self.max_sample_gap_seconds
        ]
        if len(selected) < 2:
            return None

        timestamps = np.asarray([item[0] for item in selected], dtype=np.float64)
        poses = np.stack([item[1] for item in selected], axis=0).astype(np.float32)
        observed_start = max(start_time, float(timestamps[0]))
        coverage = max(0.0, end_time - observed_start) / max(self.target_window_seconds, 1e-6)
        coverage = float(np.clip(coverage, 0.0, 1.0))
        if coverage < self.min_window_coverage:
            return None

        grid = start_time + np.arange(self.window_frames, dtype=np.float64) / self.feature_fps
        output = np.empty((self.window_frames, 33, 4), dtype=np.float32)

        # Mark long sampling gaps as invisible before feature interpolation.
        insertion = np.searchsorted(timestamps, grid, side="left")
        left = np.clip(insertion - 1, 0, len(timestamps) - 1)
        right = np.clip(insertion, 0, len(timestamps) - 1)
        nearest_distance = np.minimum(
            np.abs(grid - timestamps[left]),
            np.abs(grid - timestamps[right]),
        )
        gap_mask = nearest_distance > self.max_sample_gap_seconds

        for joint in range(33):
            for channel in range(4):
                values = poses[:, joint, channel]
                finite = np.isfinite(values)
                if not np.any(finite):
                    output[:, joint, channel] = 0.0 if channel == 3 else np.nan
                elif int(np.count_nonzero(finite)) == 1:
                    output[:, joint, channel] = float(values[finite][0])
                else:
                    output[:, joint, channel] = np.interp(
                        grid,
                        timestamps[finite],
                        values[finite],
                    ).astype(np.float32)

        output[..., 3] = np.clip(
            np.nan_to_num(output[..., 3], nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            1.0,
        )
        output[gap_mask, :, 3] = 0.0
        output[gap_mask, :, :3] = np.nan
        return output, coverage

    def predict_history(
        self,
        history: Sequence[tuple[float, np.ndarray]],
    ) -> BehaviorPrediction | None:
        prepared = self._resample_history(history)
        if prepared is None:
            return None
        landmarks, coverage = prepared
        features = extract_frame_features(landmarks, fps=self.feature_fps)
        if features.valid_ratio < self.min_valid_ratio:
            return None
        if features.names != self.feature_names:
            raise RuntimeError(
                "Runtime pose feature order differs from the trained checkpoint"
            )
        values = np.asarray(features.values, dtype=np.float32)
        if values.shape != (self.window_frames, self.input_features):
            raise RuntimeError(
                f"Unexpected behavior feature shape {values.shape}; expected "
                f"({self.window_frames}, {self.input_features})"
            )
        normalized = (values[None, ...] - self._mean) / self._std

        started = time.perf_counter()
        tensor = self._torch.from_numpy(normalized.astype(np.float32)).to(self.device)
        with self._torch.inference_mode():
            logits = self.model(tensor)
            probabilities = self._torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        inference_ms = (time.perf_counter() - started) * 1000.0

        probability_map = {
            label: float(probabilities[index])
            for index, label in enumerate(self.labels)
        }
        best_index = int(np.argmax(probabilities))
        return BehaviorPrediction(
            label=self.labels[best_index],
            confidence=float(probabilities[best_index]),
            probabilities=probability_map,
            valid_ratio=float(features.valid_ratio),
            window_seconds=float(coverage * self.target_window_seconds),
            inference_ms=float(inference_ms),
        )
