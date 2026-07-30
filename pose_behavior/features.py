from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .constants import (
    ANGLE_TRIPLETS,
    KEY_JOINT_NAMES,
    KEY_JOINTS,
    LEFT_ANKLE,
    LEFT_EAR,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    NOSE,
    POSE_LANDMARK_COUNT,
    RIGHT_ANKLE,
    RIGHT_EAR,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)

_EPS = 1e-6


@dataclass(frozen=True)
class FrameFeatureResult:
    values: np.ndarray
    names: list[str]
    valid_ratio: float


def _safe_midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.nanmean(np.stack([a, b], axis=0), axis=0)
    return out


def _interpolate_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = values.copy()
    valid = np.isfinite(out)
    if not np.any(valid):
        return np.zeros_like(out)
    idx = np.arange(len(out))
    if valid.sum() == 1:
        out[:] = out[valid][0]
        return out
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def interpolate_landmarks(landmarks: np.ndarray, visibility_threshold: float = 0.15) -> np.ndarray:
    """Fill missing pose coordinates along time while preserving visibility.

    Input shape: ``(T, 33, >=4)`` with x, y, z, visibility.
    """
    arr = np.asarray(landmarks, dtype=np.float32).copy()
    if arr.ndim != 3 or arr.shape[1] != POSE_LANDMARK_COUNT or arr.shape[2] < 4:
        raise ValueError("landmarks must have shape (T, 33, >=4)")

    low_visibility = arr[..., 3] < visibility_threshold
    arr[..., :3][low_visibility] = np.nan
    for joint in range(arr.shape[1]):
        for channel in range(3):
            arr[:, joint, channel] = _interpolate_1d(arr[:, joint, channel])
        arr[:, joint, 3] = np.nan_to_num(arr[:, joint, 3], nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    numerator = np.sum(ba * bc, axis=-1)
    denominator = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    cosine = np.clip(numerator / np.maximum(denominator, _EPS), -1.0, 1.0)
    return np.degrees(np.arccos(cosine)).astype(np.float32)


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1).astype(np.float32)


def _finite_diff(values: np.ndarray, fps: float) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values)
    return np.gradient(values, axis=0) * float(fps)


def extract_frame_features(
    landmarks: np.ndarray,
    fps: float,
    visibility_threshold: float = 0.15,
) -> FrameFeatureResult:
    """Convert MediaPipe landmarks into scale-normalized temporal features.

    The representation intentionally does **not** rotate the body to a canonical
    horizontal pose. Preserving the camera-relative vertical direction is
    important for distinguishing standing, falling and lying.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")

    raw = np.asarray(landmarks, dtype=np.float32)
    valid_ratio = float(np.mean(raw[..., 3] >= visibility_threshold))
    arr = interpolate_landmarks(raw, visibility_threshold=visibility_threshold)
    xyz = arr[..., :3]
    visibility = np.clip(arr[..., 3], 0.0, 1.0)

    hip_mid = _safe_midpoint(xyz[:, LEFT_HIP], xyz[:, RIGHT_HIP])
    shoulder_mid = _safe_midpoint(xyz[:, LEFT_SHOULDER], xyz[:, RIGHT_SHOULDER])
    mouth_mid = _safe_midpoint(xyz[:, MOUTH_LEFT], xyz[:, MOUTH_RIGHT])
    ear_mid = _safe_midpoint(xyz[:, LEFT_EAR], xyz[:, RIGHT_EAR])

    shoulder_width = _distance(xyz[:, LEFT_SHOULDER], xyz[:, RIGHT_SHOULDER])
    hip_width = _distance(xyz[:, LEFT_HIP], xyz[:, RIGHT_HIP])
    torso_length = _distance(shoulder_mid, hip_mid)
    scale = np.nanmedian(np.stack([shoulder_width, hip_width, torso_length], axis=1), axis=1)
    fallback_scale = float(np.nanmedian(scale[np.isfinite(scale) & (scale > _EPS)])) if np.any(np.isfinite(scale) & (scale > _EPS)) else 1.0
    scale = np.where(np.isfinite(scale) & (scale > _EPS), scale, fallback_scale)

    centered = (xyz - hip_mid[:, None, :]) / scale[:, None, None]

    parts: list[np.ndarray] = []
    names: list[str] = []

    for local_idx, (joint_idx, joint_name) in enumerate(zip(KEY_JOINTS, KEY_JOINT_NAMES)):
        for axis, axis_name in enumerate(("x", "y", "z")):
            parts.append(centered[:, joint_idx, axis:axis + 1])
            names.append(f"joint.{joint_name}.{axis_name}")
        parts.append(visibility[:, joint_idx:joint_idx + 1])
        names.append(f"joint.{joint_name}.visibility")

    # Camera-relative global geometry. These features retain gravity/image
    # orientation and are crucial for fall/lying recognition.
    visible_mask = visibility >= visibility_threshold
    masked_xy = np.where(visible_mask[..., None], xyz[..., :2], np.nan)
    x_min = np.nanmin(masked_xy[..., 0], axis=1)
    x_max = np.nanmax(masked_xy[..., 0], axis=1)
    y_min = np.nanmin(masked_xy[..., 1], axis=1)
    y_max = np.nanmax(masked_xy[..., 1], axis=1)
    bbox_w = np.maximum(x_max - x_min, _EPS)
    bbox_h = np.maximum(y_max - y_min, _EPS)

    torso_vec = shoulder_mid - hip_mid
    torso_angle_from_vertical = np.degrees(np.arctan2(np.abs(torso_vec[:, 0]), np.abs(torso_vec[:, 1]) + _EPS))
    shoulder_tilt = np.degrees(np.arctan2(
        xyz[:, RIGHT_SHOULDER, 1] - xyz[:, LEFT_SHOULDER, 1],
        xyz[:, RIGHT_SHOULDER, 0] - xyz[:, LEFT_SHOULDER, 0] + _EPS,
    ))
    hip_tilt = np.degrees(np.arctan2(
        xyz[:, RIGHT_HIP, 1] - xyz[:, LEFT_HIP, 1],
        xyz[:, RIGHT_HIP, 0] - xyz[:, LEFT_HIP, 0] + _EPS,
    ))

    global_features = {
        "global.hip_center_x": hip_mid[:, 0],
        "global.hip_center_y": hip_mid[:, 1],
        "global.body_scale": scale,
        "global.bbox_width": bbox_w,
        "global.bbox_height": bbox_h,
        "global.bbox_aspect": bbox_w / bbox_h,
        "global.torso_angle_from_vertical_deg": torso_angle_from_vertical,
        "global.shoulder_tilt_deg": shoulder_tilt,
        "global.hip_tilt_deg": hip_tilt,
        "global.visible_fraction": np.mean(visible_mask, axis=1),
    }
    for name, values in global_features.items():
        parts.append(np.asarray(values, dtype=np.float32)[:, None])
        names.append(name)

    # Joint angles.
    for name, (a, b, c) in ANGLE_TRIPLETS.items():
        values = _angle_deg(centered[:, a], centered[:, b], centered[:, c])
        parts.append(values[:, None])
        names.append(f"angle.{name}.deg")

    # Behavior-specific relational cues.
    relational = {
        "relation.left_wrist_to_mouth": _distance(centered[:, LEFT_WRIST], (mouth_mid - hip_mid) / scale[:, None]),
        "relation.right_wrist_to_mouth": _distance(centered[:, RIGHT_WRIST], (mouth_mid - hip_mid) / scale[:, None]),
        "relation.left_wrist_to_ear": _distance(centered[:, LEFT_WRIST], (ear_mid - hip_mid) / scale[:, None]),
        "relation.right_wrist_to_ear": _distance(centered[:, RIGHT_WRIST], (ear_mid - hip_mid) / scale[:, None]),
        "relation.ankle_distance": _distance(centered[:, LEFT_ANKLE], centered[:, RIGHT_ANKLE]),
        "relation.knee_distance": _distance(centered[:, LEFT_KNEE], centered[:, RIGHT_KNEE]),
        "relation.nose_to_hip": _distance(centered[:, NOSE], np.zeros_like(centered[:, NOSE])),
    }
    for name, values in relational.items():
        parts.append(values[:, None])
        names.append(name)

    base = np.concatenate(parts, axis=1).astype(np.float32)

    velocity = _finite_diff(base, fps)
    acceleration = _finite_diff(velocity, fps)
    velocity_names = [f"vel.{name}" for name in names]
    acceleration_names = [f"acc.{name}" for name in names]

    output = np.concatenate([base, velocity, acceleration], axis=1)
    output_names = names + velocity_names + acceleration_names
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return FrameFeatureResult(output, output_names, valid_ratio)


def _slope(values: np.ndarray) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros(values.shape[1], dtype=np.float32)
    x = np.linspace(-1.0, 1.0, values.shape[0], dtype=np.float32)
    x_centered = x - x.mean()
    denom = float(np.sum(x_centered ** 2))
    return ((x_centered[:, None] * (values - values.mean(axis=0, keepdims=True))).sum(axis=0) / max(denom, _EPS)).astype(np.float32)


def aggregate_window_features(
    frame_features: np.ndarray,
    frame_feature_names: Iterable[str],
) -> tuple[np.ndarray, list[str]]:
    """Aggregate a variable-length sequence for a tree-based classifier."""
    values = np.asarray(frame_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("frame_features must have shape (T, F), T > 0")
    names = list(frame_feature_names)
    if values.shape[1] != len(names):
        raise ValueError("feature name count does not match frame feature width")

    stats = {
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "min": np.min(values, axis=0),
        "max": np.max(values, axis=0),
        "p10": np.percentile(values, 10, axis=0),
        "median": np.percentile(values, 50, axis=0),
        "p90": np.percentile(values, 90, axis=0),
        "range": np.max(values, axis=0) - np.min(values, axis=0),
        "delta": values[-1] - values[0],
        "slope": _slope(values),
    }
    out = np.concatenate([stats[key] for key in stats], axis=0).astype(np.float32)
    out_names = [f"{stat}.{feature}" for stat in stats for feature in names]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), out_names
