from __future__ import annotations

import math
from typing import Iterable
import numpy as np

LANDMARK_COUNT = 33
POSITION_DIM = 99
VELOCITY_START = 99
ACCELERATION_START = 198
CONFIDENCE_START = 297
OBSERVED_START = 330
GLOBAL_START = 363
GLOBAL_DIM = 24
FEATURE_DIM = GLOBAL_START + GLOBAL_DIM

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

HEAD = tuple(range(0, 11))
TORSO = (11, 12, 23, 24)
LEFT_ARM = (11, 13, 15, 17, 19, 21)
RIGHT_ARM = (12, 14, 16, 18, 20, 22)
LEFT_LEG = (23, 25, 27, 29, 31)
RIGHT_LEG = (24, 26, 28, 30, 32)

GLOBAL_INDEX = {
    "hip_x": 0,
    "hip_y": 1,
    "shoulder_x": 2,
    "shoulder_y": 3,
    "bbox_w": 4,
    "bbox_h": 5,
    "bbox_aspect": 6,
    "bbox_cx": 7,
    "bbox_cy": 8,
    "torso_sin": 9,
    "torso_cos": 10,
    "shoulder_sin": 11,
    "shoulder_cos": 12,
    "hip_sin": 13,
    "hip_cos": 14,
    "left_knee": 15,
    "right_knee": 16,
    "mean_confidence": 17,
    "visible_ratio": 18,
    "torso_coverage": 19,
    "left_arm_coverage": 20,
    "right_arm_coverage": 21,
    "left_leg_coverage": 22,
    "right_leg_coverage": 23,
}


def _safe_norm(v: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    return np.sqrt(np.maximum(np.sum(v * v, axis=axis, keepdims=keepdims), 1e-12))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    cos = np.sum(ba * bc, axis=-1) / (_safe_norm(ba) * _safe_norm(bc))
    return np.arccos(np.clip(cos, -1.0, 1.0)) / math.pi


def effective_confidence(poses: np.ndarray) -> np.ndarray:
    """
    MediaPipe usually returns all 33 landmarks. Occluded points are represented
    by low visibility/presence rather than by a shorter list.
    """
    arr = np.asarray(poses, dtype=np.float32)
    visibility = np.clip(arr[..., 3], 0.0, 1.0)
    if arr.shape[2] >= 5:
        presence = np.clip(arr[..., 4], 0.0, 1.0)
        confidence = np.minimum(visibility, presence)
    else:
        confidence = visibility
    confidence[~np.isfinite(confidence)] = 0.0
    return confidence


def interpolate_missing_limited(
    poses: np.ndarray,
    min_confidence: float = 0.20,
    max_gap_frames: int = 6,
    edge_hold_frames: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate only short gaps. Long occlusions are not hallucinated.

    Returns:
      xyz: filled T×33×3
      observed: actually observed landmark mask
      usable: observed or safely interpolated mask
      confidence: MediaPipe effective confidence
    """
    arr = np.asarray(poses, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != LANDMARK_COUNT or arr.shape[2] < 4:
        raise ValueError(f"Expected T×33×4/5, got {arr.shape}")

    raw_xyz = arr[..., :3]
    confidence = effective_confidence(arr)
    observed = np.isfinite(raw_xyz).all(axis=2) & (confidence >= min_confidence)
    xyz = np.zeros_like(raw_xyz, dtype=np.float32)
    usable = observed.copy()
    xyz[observed] = raw_xyz[observed]

    for landmark in range(LANDMARK_COUNT):
        valid_idx = np.flatnonzero(observed[:, landmark])
        if len(valid_idx) == 0:
            continue

        # Hold a reliable endpoint only briefly.
        first, last = int(valid_idx[0]), int(valid_idx[-1])
        for frame in range(max(0, first - edge_hold_frames), first):
            xyz[frame, landmark] = raw_xyz[first, landmark]
            usable[frame, landmark] = True
        for frame in range(last + 1, min(len(arr), last + edge_hold_frames + 1)):
            xyz[frame, landmark] = raw_xyz[last, landmark]
            usable[frame, landmark] = True

        # Interpolate only short internal gaps.
        for left, right in zip(valid_idx[:-1], valid_idx[1:]):
            gap = int(right - left - 1)
            if gap <= 0 or gap > max_gap_frames:
                continue
            alpha = np.arange(1, gap + 1, dtype=np.float32) / float(gap + 1)
            xyz[left + 1:right, landmark] = (
                raw_xyz[left, landmark][None, :] * (1.0 - alpha[:, None])
                + raw_xyz[right, landmark][None, :] * alpha[:, None]
            )
            usable[left + 1:right, landmark] = True

    xyz[~np.isfinite(xyz)] = 0.0
    return xyz, observed.astype(np.float32), usable.astype(np.float32), confidence


def resample_pose_sequence(
    timestamps: Iterable[float],
    poses: np.ndarray,
    target_fps: float,
    frame_count: int,
    end_time: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample irregular per-track samples to a fixed past-only time grid."""
    times = np.asarray(list(timestamps), dtype=np.float64)
    arr = np.asarray(poses, dtype=np.float32)
    if len(times) != len(arr):
        raise ValueError("timestamps and poses length differ")
    if len(times) < 2:
        raise ValueError("At least 2 pose samples are required")
    order = np.argsort(times)
    times, arr = times[order], arr[order]
    unique = np.concatenate(([True], np.diff(times) > 1e-6))
    times, arr = times[unique], arr[unique]
    if end_time is None:
        end_time = float(times[-1])
    duration = (frame_count - 1) / float(target_fps)
    grid = np.linspace(end_time - duration, end_time, frame_count, dtype=np.float64)

    output = np.zeros((frame_count, LANDMARK_COUNT, arr.shape[2]), dtype=np.float32)
    available = (grid >= times[0]) & (grid <= times[-1])
    for j in range(LANDMARK_COUNT):
        for k in range(arr.shape[2]):
            values = arr[:, j, k]
            finite = np.isfinite(values)
            if finite.sum() >= 2:
                output[:, j, k] = np.interp(grid, times[finite], values[finite]).astype(np.float32)
            elif finite.sum() == 1:
                output[:, j, k] = float(values[finite][0])
    return output, available.astype(np.float32), grid


def _mean_points(xyz: np.ndarray, usable: np.ndarray, indices: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    points = xyz[:, indices]
    mask = usable[:, indices].astype(bool)
    count = mask.sum(axis=1)
    summed = np.sum(points * mask[..., None], axis=1)
    center = summed / np.maximum(count[:, None], 1)
    return center.astype(np.float32), (count > 0)


def _body_coverage(observed: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    return np.mean(observed[:, indices], axis=1).astype(np.float32)


def _robust_centers_and_scale(xyz: np.ndarray, usable: np.ndarray):
    hip_mid, hip_ok = _mean_points(xyz, usable, (LEFT_HIP, RIGHT_HIP))
    shoulder_mid, shoulder_ok = _mean_points(xyz, usable, (LEFT_SHOULDER, RIGHT_SHOULDER))
    all_mid, all_ok = _mean_points(xyz, usable, tuple(range(LANDMARK_COUNT)))

    center = hip_mid.copy()
    center[~hip_ok & shoulder_ok] = shoulder_mid[~hip_ok & shoulder_ok]
    center[~hip_ok & ~shoulder_ok & all_ok] = all_mid[~hip_ok & ~shoulder_ok & all_ok]

    shoulder_pair = usable[:, LEFT_SHOULDER].astype(bool) & usable[:, RIGHT_SHOULDER].astype(bool)
    hip_pair = usable[:, LEFT_HIP].astype(bool) & usable[:, RIGHT_HIP].astype(bool)
    shoulder_width = _safe_norm(xyz[:, LEFT_SHOULDER] - xyz[:, RIGHT_SHOULDER])
    hip_width = _safe_norm(xyz[:, LEFT_HIP] - xyz[:, RIGHT_HIP])
    torso_len = _safe_norm(shoulder_mid - hip_mid)
    torso_ok = shoulder_ok & hip_ok

    candidates = np.stack(
        [
            np.where(shoulder_pair, shoulder_width, np.nan),
            np.where(hip_pair, hip_width, np.nan),
            np.where(torso_ok, torso_len, np.nan),
        ],
        axis=1,
    )
    finite_candidates = np.where(np.isfinite(candidates), candidates, -np.inf)
    raw_scale = np.max(finite_candidates, axis=1)
    raw_scale[raw_scale == -np.inf] = np.nan
    good_scale = np.isfinite(raw_scale) & (raw_scale > 1e-3)
    fallback = float(np.nanmedian(raw_scale[good_scale])) if np.any(good_scale) else 0.20
    scale = np.where(good_scale, raw_scale, fallback).astype(np.float32)
    scale = np.clip(scale, 1e-3, 2.0)
    return center, hip_mid, shoulder_mid, hip_ok, shoulder_ok, scale


def extract_frame_features(
    poses: np.ndarray,
    fps: float = 15.0,
    min_visibility: float = 0.20,
    max_interpolation_gap_frames: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build T×387 input with explicit confidence and observed masks.

    Missing limbs are represented as neutral coordinates plus confidence/mask=0.
    Short gaps are interpolated; long gaps stay missing.
    """
    arr = np.asarray(poses, dtype=np.float32)
    xyz, observed, usable, confidence = interpolate_missing_limited(
        arr,
        min_confidence=min_visibility,
        max_gap_frames=max_interpolation_gap_frames,
    )
    center, hip_mid, shoulder_mid, hip_ok, shoulder_ok, scale = _robust_centers_and_scale(xyz, usable)

    relative = (xyz - center[:, None, :]) / scale[:, None, None]
    relative = np.clip(relative, -4.0, 4.0)
    relative *= usable[..., None]

    velocity = np.gradient(relative, axis=0) * float(fps)
    motion_mask = usable.astype(bool)
    motion_mask[1:] &= usable[:-1].astype(bool)
    motion_mask[0] = False
    velocity *= motion_mask[..., None]
    acceleration = np.gradient(velocity, axis=0) * float(fps)
    accel_mask = motion_mask.copy()
    accel_mask[1:] &= motion_mask[:-1]
    accel_mask[0] = False
    acceleration *= accel_mask[..., None]
    velocity = np.clip(velocity, -20.0, 20.0)
    acceleration = np.clip(acceleration, -100.0, 100.0)

    bbox_w = np.zeros(len(arr), dtype=np.float32)
    bbox_h = np.zeros(len(arr), dtype=np.float32)
    bbox_cx = np.zeros(len(arr), dtype=np.float32)
    bbox_cy = np.zeros(len(arr), dtype=np.float32)
    for i in range(len(arr)):
        mask = observed[i].astype(bool)
        if mask.sum() < 2:
            mask = usable[i].astype(bool)
        if mask.sum() >= 2:
            points = xyz[i, mask, :2]
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            bbox_w[i] = max(float(x_max - x_min), 1e-4)
            bbox_h[i] = max(float(y_max - y_min), 1e-4)
            bbox_cx[i] = float((x_min + x_max) * 0.5)
            bbox_cy[i] = float((y_min + y_max) * 0.5)
    bbox_aspect = np.clip(bbox_w / np.maximum(bbox_h, 1e-4), 0.0, 10.0)

    torso = shoulder_mid[:, :2] - hip_mid[:, :2]
    torso_angle = np.where(
        shoulder_ok & hip_ok,
        np.arctan2(torso[:, 0], -torso[:, 1]),
        0.0,
    )
    shoulder_pair = usable[:, LEFT_SHOULDER].astype(bool) & usable[:, RIGHT_SHOULDER].astype(bool)
    shoulder = xyz[:, RIGHT_SHOULDER, :2] - xyz[:, LEFT_SHOULDER, :2]
    shoulder_angle = np.where(shoulder_pair, np.arctan2(shoulder[:, 1], shoulder[:, 0]), 0.0)
    hip_pair = usable[:, LEFT_HIP].astype(bool) & usable[:, RIGHT_HIP].astype(bool)
    hipline = xyz[:, RIGHT_HIP, :2] - xyz[:, LEFT_HIP, :2]
    hip_angle = np.where(hip_pair, np.arctan2(hipline[:, 1], hipline[:, 0]), 0.0)

    left_knee_ok = (
        usable[:, LEFT_HIP].astype(bool)
        & usable[:, LEFT_KNEE].astype(bool)
        & usable[:, LEFT_ANKLE].astype(bool)
    )
    right_knee_ok = (
        usable[:, RIGHT_HIP].astype(bool)
        & usable[:, RIGHT_KNEE].astype(bool)
        & usable[:, RIGHT_ANKLE].astype(bool)
    )
    left_knee = np.where(
        left_knee_ok,
        _angle(xyz[:, LEFT_HIP], xyz[:, LEFT_KNEE], xyz[:, LEFT_ANKLE]),
        0.0,
    )
    right_knee = np.where(
        right_knee_ok,
        _angle(xyz[:, RIGHT_HIP], xyz[:, RIGHT_KNEE], xyz[:, RIGHT_ANKLE]),
        0.0,
    )

    mean_confidence = np.mean(confidence, axis=1).astype(np.float32)
    visible_ratio = np.mean(observed, axis=1).astype(np.float32)
    torso_coverage = _body_coverage(observed, TORSO)
    left_arm_coverage = _body_coverage(observed, LEFT_ARM)
    right_arm_coverage = _body_coverage(observed, RIGHT_ARM)
    left_leg_coverage = _body_coverage(observed, LEFT_LEG)
    right_leg_coverage = _body_coverage(observed, RIGHT_LEG)

    globals_ = np.stack(
        [
            hip_mid[:, 0], hip_mid[:, 1],
            shoulder_mid[:, 0], shoulder_mid[:, 1],
            bbox_w, bbox_h, bbox_aspect, bbox_cx, bbox_cy,
            np.sin(torso_angle), np.cos(torso_angle),
            np.sin(shoulder_angle), np.cos(shoulder_angle),
            np.sin(hip_angle), np.cos(hip_angle),
            left_knee, right_knee,
            mean_confidence, visible_ratio, torso_coverage,
            left_arm_coverage, right_arm_coverage,
            left_leg_coverage, right_leg_coverage,
        ],
        axis=1,
    ).astype(np.float32)

    features = np.concatenate(
        [
            relative.reshape(len(arr), -1),
            velocity.reshape(len(arr), -1),
            acceleration.reshape(len(arr), -1),
            confidence,
            observed,
            globals_,
        ],
        axis=1,
    ).astype(np.float32)
    if features.shape[1] != FEATURE_DIM:
        raise AssertionError(f"Feature dimension {features.shape[1]} != {FEATURE_DIM}")

    # Torso-only still supports fall/ground inference. Lower-body-dependent
    # unstable gait is gated separately at runtime.
    quality = (
        0.65 * torso_coverage
        + 0.25 * visible_ratio
        + 0.10 * mean_confidence
    ).astype(np.float32)
    features[~np.isfinite(features)] = 0.0
    return features, quality


def coverage_from_features(features: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(features)
    g = x[..., GLOBAL_START:GLOBAL_START + GLOBAL_DIM]
    return {
        name: g[..., index]
        for name, index in GLOBAL_INDEX.items()
        if name.endswith("_coverage") or name in {"visible_ratio", "mean_confidence"}
    }


def _zero_landmarks(out: np.ndarray, frames: slice, landmarks: tuple[int, ...]) -> None:
    for landmark in landmarks:
        for base in (0, VELOCITY_START, ACCELERATION_START):
            start = base + landmark * 3
            out[frames, start:start + 3] = 0.0
        out[frames, CONFIDENCE_START + landmark] = 0.0
        out[frames, OBSERVED_START + landmark] = 0.0


def augment_features(
    x: np.ndarray,
    rng: np.random.Generator,
    jitter_std: float = 0.012,
    body_part_dropout_prob: float = 0.55,
    torso_only_prob: float = 0.18,
    frame_dropout_prob: float = 0.25,
) -> np.ndarray:
    """
    Occlusion-aware augmentation:
      - short/long loss of a whole arm or leg,
      - occasional torso-only interval,
      - complete pose loss for a few frames,
      - jitter only on continuous features, never on confidence/masks.
    """
    out = np.asarray(x, dtype=np.float32).copy()
    length = len(out)

    out[:, :POSITION_DIM] += rng.normal(0.0, jitter_std, size=(length, POSITION_DIM)).astype(np.float32)
    out[:, VELOCITY_START:ACCELERATION_START] += rng.normal(
        0.0, jitter_std * 2.0, size=(length, POSITION_DIM)
    ).astype(np.float32)
    out[:, ACCELERATION_START:CONFIDENCE_START] += rng.normal(
        0.0, jitter_std * 4.0, size=(length, POSITION_DIM)
    ).astype(np.float32)

    if rng.random() < body_part_dropout_prob:
        part_name, landmarks, global_name = [
            ("left_arm", LEFT_ARM, "left_arm_coverage"),
            ("right_arm", RIGHT_ARM, "right_arm_coverage"),
            ("left_leg", LEFT_LEG, "left_leg_coverage"),
            ("right_leg", RIGHT_LEG, "right_leg_coverage"),
            ("head", HEAD, None),
        ][int(rng.integers(0, 5))]
        span = int(rng.integers(max(3, length // 8), max(4, length + 1)))
        start = int(rng.integers(0, max(1, length - span + 1)))
        frames = slice(start, min(length, start + span))
        _zero_landmarks(out, frames, landmarks)
        if global_name is not None:
            out[frames, GLOBAL_START + GLOBAL_INDEX[global_name]] = 0.0

    if rng.random() < torso_only_prob:
        span = int(rng.integers(max(4, length // 6), max(5, length // 2 + 1)))
        start = int(rng.integers(0, max(1, length - span + 1)))
        frames = slice(start, min(length, start + span))
        hidden = tuple(sorted(set(range(LANDMARK_COUNT)).difference(TORSO)))
        _zero_landmarks(out, frames, hidden)
        out[frames, GLOBAL_START + GLOBAL_INDEX["left_arm_coverage"]] = 0.0
        out[frames, GLOBAL_START + GLOBAL_INDEX["right_arm_coverage"]] = 0.0
        out[frames, GLOBAL_START + GLOBAL_INDEX["left_leg_coverage"]] = 0.0
        out[frames, GLOBAL_START + GLOBAL_INDEX["right_leg_coverage"]] = 0.0
        out[frames, GLOBAL_START + GLOBAL_INDEX["visible_ratio"]] = len(TORSO) / LANDMARK_COUNT

    if rng.random() < frame_dropout_prob and length >= 8:
        width = int(rng.integers(2, max(3, length // 7)))
        start = int(rng.integers(0, length - width + 1))
        out[start:start + width] = 0.0

    # Masks remain binary after augmentation, then coverage globals are
    # recomputed so the model never receives contradictory mask/coverage data.
    observed = (
        out[:, OBSERVED_START:GLOBAL_START] > 0.5
    ).astype(np.float32)
    out[:, OBSERVED_START:GLOBAL_START] = observed
    confidence = np.clip(
        out[:, CONFIDENCE_START:OBSERVED_START], 0.0, 1.0
    )
    out[:, CONFIDENCE_START:OBSERVED_START] = confidence

    out[:, GLOBAL_START + GLOBAL_INDEX["mean_confidence"]] = np.mean(confidence, axis=1)
    out[:, GLOBAL_START + GLOBAL_INDEX["visible_ratio"]] = np.mean(observed, axis=1)
    for indices, key in (
        (TORSO, "torso_coverage"),
        (LEFT_ARM, "left_arm_coverage"),
        (RIGHT_ARM, "right_arm_coverage"),
        (LEFT_LEG, "left_leg_coverage"),
        (RIGHT_LEG, "right_leg_coverage"),
    ):
        out[:, GLOBAL_START + GLOBAL_INDEX[key]] = np.mean(observed[:, indices], axis=1)

    out[~np.isfinite(out)] = 0.0
    return out
