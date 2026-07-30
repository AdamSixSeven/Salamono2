"""Lightweight temporal posture / coordination anomaly analysis.

The module deliberately detects *observable motion patterns* (body sway,
trajectory instability, irregular stepping, sudden balance loss).  It does not
infer intoxication, a medical condition, or a cause of the movement.

Pipeline:
    existing YOLO person boxes -> MediaPipe Pose Landmarker Lite ->
    lightweight track matching -> 2-4 s temporal features -> risk score

MediaPipe is imported lazily so the rest of Perimetr can still start when the
optional dependency or model bundle is unavailable.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
import importlib.util
import math
import os
import threading
import time
from typing import Callable, Protocol, TYPE_CHECKING

import cv2
import numpy as np

from backend.detector import Detection
from config import CONFIG, PostureConfig

if TYPE_CHECKING:
    from backend.behavior_classifier import BehaviorClassifier


# MediaPipe Pose Landmarker returns the 33-landmark BlazePose topology.
POSE_LANDMARK_COUNT = 33
POSTURE_DRAW_MIN_VISIBILITY = 0.50
PERSON_CROP_MARGIN = 0.24

# Landmark indices used by the posture heuristics.
LEFT_MOUTH = 9
RIGHT_MOUTH = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# Full topology published by MediaPipe Tasks as
# PoseLandmarksConnections.POSE_LANDMARKS.  Keep a local copy so importing this
# module still works when the optional mediapipe package is unavailable.
POSTURE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # Face.
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (LEFT_MOUTH, RIGHT_MOUTH),
    # Torso and arms, including the hand landmarks emitted by Pose Landmarker.
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (RIGHT_SHOULDER, 14), (14, 16),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    # Legs and feet.
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL), (LEFT_HEEL, LEFT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_HEEL), (RIGHT_HEEL, RIGHT_FOOT_INDEX),
    (LEFT_ANKLE, LEFT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
)


class PoseEstimator(Protocol):
    """Provider interface used by the analyzer and by deterministic tests."""

    def estimate(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[np.ndarray]:
        """Return zero or more arrays shaped (33, 4): x, y, z, visibility."""

    def close(self) -> None:
        """Release native resources."""


class MediaPipePoseEstimator:
    """MediaPipe Tasks Pose Landmarker running in VIDEO mode.

    One estimator instance is kept per camera.  VIDEO mode lets MediaPipe reuse
    tracking state between samples and avoid a full detector pass on every
    frame, which is important for the CPU-oriented MVP.
    """

    def __init__(self, config: PostureConfig):
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(
                f"MediaPipe pose model not found: {config.model_path}. "
                "Run: python tools/download_models.py pose"
            )

        import mediapipe as mp  # lazy optional import

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=config.model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            # PostureAnalyzer feeds one YOLO person crop at a time.  Keeping
            # this at one avoids asking MediaPipe to search a crop for poses
            # that cannot be used; ``config.max_poses`` limits the number of
            # largest person crops analyzed per sample.
            num_poses=1,
            min_pose_detection_confidence=config.min_pose_detection_confidence,
            min_pose_presence_confidence=config.min_pose_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._mp = mp
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def estimate(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[np.ndarray]:
        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=frame_rgb,
        )
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        poses: list[np.ndarray] = []
        for pose in result.pose_landmarks:
            arr = np.asarray(
                [[p.x, p.y, p.z, getattr(p, "visibility", 1.0)] for p in pose],
                dtype=np.float32,
            )
            if arr.shape == (33, 4):
                poses.append(arr)
        return poses

    def close(self) -> None:
        self._landmarker.close()


@dataclass
class PoseSample:
    timestamp: float
    torso_angle_deg: float
    shoulder_tilt_deg: float
    hip_x: float
    hip_y: float
    bbox_width: float
    bbox_height: float
    ankle_separation: float | None
    hand_to_mouth: bool | None
    pose_confidence: float


@dataclass
class PostureAssessment:
    track_id: int
    person: Detection
    risk_score: float
    severity: str
    status: str
    signals: list[str]
    metrics: dict[str, float]
    frame_timestamp: float
    pose_confidence: float
    history_seconds: float
    confirmed: bool = False
    landmarks: np.ndarray | None = None
    # Dimensions of the frame in which normalized landmarks were produced.
    # They let cached landmarks remain correct if the stream dimensions
    # change.  Defaults preserve compatibility with manually-created
    # assessments in routes and tests.
    frame_width: int | None = None
    frame_height: int | None = None
    behavior_label: str | None = None
    behavior_confidence: float = 0.0
    behavior_probabilities: dict[str, float] = field(default_factory=dict)
    behavior_valid_ratio: float = 0.0
    behavior_window_seconds: float = 0.0
    behavior_inference_ms: float = 0.0


@dataclass
class _Track:
    track_id: int
    last_box: tuple[int, int, int, int]
    last_seen: float
    history: deque[PoseSample] = field(default_factory=deque)
    alert_streak: int = 0
    last_alert_time: float = float("-inf")
    fall_streak: int = 0
    last_fall_alert_time: float = float("-inf")
    gesture_streak: int = 0
    last_gesture_alert_time: float = float("-inf")
    latest: PostureAssessment | None = None
    roi_box: tuple[int, int, int, int] | None = None
    display_landmarks: np.ndarray | None = None
    display_box: tuple[int, int, int, int] | None = None
    display_frame_width: int | None = None
    display_frame_height: int | None = None
    behavior_history: deque[tuple[float, np.ndarray]] = field(default_factory=deque)
    behavior_probability_history: deque[dict[str, float]] = field(default_factory=deque)
    behavior_samples_since_inference: int = 0
    behavior_prediction_version: int = 0
    behavior_streak_version: int = 0
    behavior_fall_streak: int = 0
    behavior_lying_streak: int = 0
    behavior_last_alert_time: float = float("-inf")
    behavior_label: str | None = None
    behavior_confidence: float = 0.0
    behavior_probabilities: dict[str, float] = field(default_factory=dict)
    behavior_valid_ratio: float = 0.0
    behavior_window_seconds: float = 0.0
    behavior_inference_ms: float = 0.0


@dataclass
class PostureProcessResult:
    assessments: list[PostureAssessment]
    inference_ran: bool


@dataclass(frozen=True)
class PostureWorkerSnapshot:
    """Latest completed background posture result for one camera."""

    camera_id: str
    frame_timestamp: float
    completed_at: float
    result: PostureProcessResult | None
    error: str | None = None


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _box_center_distance(a: tuple[int, int, int, int],
                         b: tuple[int, int, int, int]) -> float:
    acx = (a[0] + a[2]) / 2.0
    acy = (a[1] + a[3]) / 2.0
    bcx = (b[0] + b[2]) / 2.0
    bcy = (b[1] + b[3]) / 2.0
    distance = math.hypot(acx - bcx, acy - bcy)
    scale = max(1.0, math.hypot(b[2] - b[0], b[3] - b[1]))
    return distance / scale


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _component(value: float, start: float, full: float) -> float:
    if full <= start:
        return 1.0 if value >= full else 0.0
    return _clip01((value - start) / (full - start))


def _smooth(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return values
    out = values.copy()
    out[1:-1] = (values[:-2] + values[1:-1] + values[2:]) / 3.0
    return out


def _pose_bbox(pose: np.ndarray, min_visibility: float) -> tuple[float, float, float, float] | None:
    visible = pose[:, 3] >= min_visibility
    if int(np.count_nonzero(visible)) < 6:
        return None
    xy = pose[visible, :2]
    return (float(np.min(xy[:, 0])), float(np.min(xy[:, 1])),
            float(np.max(xy[:, 0])), float(np.max(xy[:, 1])))


def _normalized_box(box: tuple[int, int, int, int], frame_w: int,
                    frame_h: int) -> tuple[float, float, float, float]:
    return (box[0] / frame_w, box[1] / frame_h,
            box[2] / frame_w, box[3] / frame_h)


def _normalized_iou(a: tuple[float, float, float, float],
                    b: tuple[float, float, float, float]) -> float:
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def stabilized_square_crop_box(
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    *,
    previous: tuple[int, int, int, int] | None = None,
    margin: float = PERSON_CROP_MARGIN,
    center_follow: float = 0.72,
    size_follow: float = 0.28,
) -> tuple[int, int, int, int] | None:
    """Build a square, padded ROI that follows position faster than scale."""
    if frame_width <= 0 or frame_height <= 0:
        return None
    x1, y1, x2, y2 = (float(value) for value in box)
    if not np.isfinite((x1, y1, x2, y2)).all() or x2 <= x1 or y2 <= y1:
        return None
    width = x2 - x1
    height = y2 - y1
    raw_cx = (x1 + x2) / 2.0
    raw_cy = (y1 + y2) / 2.0
    raw_size = max(width, height) * (1.0 + 2.0 * max(0.0, float(margin)))
    raw_size = max(8.0, raw_size)

    if previous is not None:
        px1, py1, px2, py2 = previous
        previous_size = max(float(px2 - px1), float(py2 - py1), 8.0)
        previous_cx = (px1 + px2) / 2.0
        previous_cy = (py1 + py2) / 2.0
        center_alpha = float(np.clip(center_follow, 0.0, 1.0))
        size_alpha = float(np.clip(size_follow, 0.0, 1.0))
        raw_cx = previous_cx * (1.0 - center_alpha) + raw_cx * center_alpha
        raw_cy = previous_cy * (1.0 - center_alpha) + raw_cy * center_alpha
        raw_size = previous_size * (1.0 - size_alpha) + raw_size * size_alpha

    half = raw_size / 2.0
    crop_x1 = int(math.floor(raw_cx - half))
    crop_y1 = int(math.floor(raw_cy - half))
    crop_x2 = int(math.ceil(raw_cx + half))
    crop_y2 = int(math.ceil(raw_cy + half))

    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0
    if crop_x2 > frame_width:
        shift = crop_x2 - frame_width
        crop_x1 -= shift
        crop_x2 = frame_width
    if crop_y2 > frame_height:
        shift = crop_y2 - frame_height
        crop_y1 -= shift
        crop_y2 = frame_height
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None
    return crop_x1, crop_y1, crop_x2, crop_y2


def crop_from_box(
    frame: np.ndarray,
    crop_box: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    if crop_box is None or frame.ndim < 2:
        return None
    x1, y1, x2, y2 = crop_box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop, crop_box


def constrain_display_bone_lengths(
    reference: np.ndarray,
    candidate: np.ndarray,
    fallback: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    min_visibility: float,
    min_ratio: float = 0.25,
    max_ratio: float = 1.80,
    min_change_px: float = 10.0,
) -> tuple[np.ndarray, int]:
    """Reject obvious one-frame limb stretching from display-only tracking.

    Optical flow follows landmarks independently.  A hand can occasionally
    attach to a sleeve edge or background feature and make one arm several
    times longer for one frame.  For a physically connected bone, compare its
    projected length with the previous display pose and revert the endpoint
    that deviated most from the translation-only fallback when the change is
    clearly implausible.  This affects only the overlay, never posture alerts.
    """
    out = np.array(candidate, dtype=np.float32, copy=True)
    reference = np.asarray(reference, dtype=np.float32)
    fallback = np.asarray(fallback, dtype=np.float32)
    if (
        out.shape != reference.shape
        or out.shape != fallback.shape
        or out.ndim != 2
        or out.shape[1] < 4
        or frame_width <= 0
        or frame_height <= 0
    ):
        return out, 0

    scale = np.asarray([frame_width, frame_height], dtype=np.float32)
    reverted: set[int] = set()
    for _ in range(2):
        changed = False
        for left, right in POSTURE_CONNECTIONS:
            if max(left, right) >= len(out):
                continue
            if (
                reference[left, 3] < min_visibility
                or reference[right, 3] < min_visibility
                or not np.isfinite(reference[[left, right], :2]).all()
                or not np.isfinite(out[[left, right], :2]).all()
            ):
                continue
            ref_length = float(np.linalg.norm((reference[left, :2] - reference[right, :2]) * scale))
            new_length = float(np.linalg.norm((out[left, :2] - out[right, :2]) * scale))
            if ref_length < 3.0 or not math.isfinite(new_length):
                continue
            ratio = new_length / ref_length
            if min_ratio <= ratio <= max_ratio or abs(new_length - ref_length) < min_change_px:
                continue
            left_shift = float(np.linalg.norm((out[left, :2] - fallback[left, :2]) * scale))
            right_shift = float(np.linalg.norm((out[right, :2] - fallback[right, :2]) * scale))
            index = left if left_shift >= right_shift else right
            out[index, :3] = fallback[index, :3]
            reverted.add(index)
            changed = True
        if not changed:
            break
    return out, len(reverted)


def track_pose_optical_flow(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    landmarks: np.ndarray,
    fallback: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    bbox: tuple[int, int, int, int],
    min_visibility: float,
    win_size: int = 21,
    max_level: int = 3,
    fb_threshold_px: float = 1.5,
    max_jump_frac: float = 0.18,
) -> tuple[np.ndarray, int]:
    """Track visible pose landmarks one frame forward using LK flow."""
    if (
        previous_gray.shape != current_gray.shape
        or landmarks.ndim != 2
        or landmarks.shape[1] < 4
        or frame_width <= 0
        or frame_height <= 0
    ):
        return np.array(fallback, copy=True), 0
    # LK flow on textureless frames can report numerically valid but arbitrary
    # motion. In that case the deterministic person-box transform is safer.
    if float(np.std(previous_gray)) < 1e-3 or float(np.std(current_gray)) < 1e-3:
        return np.array(fallback, copy=True), 0
    visible = (
        np.isfinite(landmarks[:, :2]).all(axis=1)
        & np.isfinite(landmarks[:, 3])
        & (landmarks[:, 3] >= min_visibility)
        & (landmarks[:, 0] >= 0.0)
        & (landmarks[:, 0] <= 1.0)
        & (landmarks[:, 1] >= 0.0)
        & (landmarks[:, 1] <= 1.0)
    )
    indexes = np.flatnonzero(visible)
    if indexes.size < 6:
        return np.array(fallback, copy=True), 0
    points = np.column_stack([
        landmarks[indexes, 0] * frame_width,
        landmarks[indexes, 1] * frame_height,
    ]).astype(np.float32).reshape(-1, 1, 2)
    win = max(5, int(win_size))
    if win % 2 == 0:
        win += 1
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
    next_points, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, points, None,
        winSize=(win, win), maxLevel=max(0, int(max_level)), criteria=criteria,
    )
    if next_points is None or status_forward is None:
        return np.array(fallback, copy=True), 0
    back_points, status_back, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, next_points, None,
        winSize=(win, win), maxLevel=max(0, int(max_level)), criteria=criteria,
    )
    if back_points is None or status_back is None:
        return np.array(fallback, copy=True), 0

    forward = next_points.reshape(-1, 2)
    backward = back_points.reshape(-1, 2)
    original = points.reshape(-1, 2)
    fb_error = np.linalg.norm(backward - original, axis=1)
    jump = np.linalg.norm(forward - original, axis=1)
    box_diag = max(1.0, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    good = (
        (status_forward.reshape(-1) > 0)
        & (status_back.reshape(-1) > 0)
        & np.isfinite(forward).all(axis=1)
        & (fb_error <= max(0.1, float(fb_threshold_px)))
        & (jump <= max(1.0, float(max_jump_frac) * box_diag))
        & (forward[:, 0] >= 0.0)
        & (forward[:, 0] < frame_width)
        & (forward[:, 1] >= 0.0)
        & (forward[:, 1] < frame_height)
    )
    tracked = np.array(fallback, dtype=np.float32, copy=True)
    good_indexes = indexes[good]
    good_points = forward[good]
    if good_indexes.size:
        tracked[good_indexes, 0] = good_points[:, 0] / frame_width
        tracked[good_indexes, 1] = good_points[:, 1] / frame_height
    tracked, reverted = constrain_display_bone_lengths(
        landmarks,
        tracked,
        fallback,
        frame_width=frame_width,
        frame_height=frame_height,
        min_visibility=min_visibility,
    )
    return tracked, max(0, int(good_indexes.size) - reverted)


def person_crop(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    margin: float = PERSON_CROP_MARGIN,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Return a clipped person crop and its full-frame pixel bounds.

    The margin is relative to the YOLO box width/height.  ``None`` is returned
    for malformed/empty frames or boxes, rather than passing an empty image to
    a native MediaPipe call.
    """

    if frame.ndim < 2:
        return None
    frame_h, frame_w = frame.shape[:2]
    if frame_w <= 0 or frame_h <= 0 or len(box) != 4:
        return None

    x1, y1, x2, y2 = (float(value) for value in box)
    if not np.isfinite((x1, y1, x2, y2)).all() or x2 <= x1 or y2 <= y1:
        return None

    safe_margin = max(0.0, float(margin))
    pad_x = (x2 - x1) * safe_margin
    pad_y = (y2 - y1) * safe_margin
    crop_x1 = max(0, int(math.floor(x1 - pad_x)))
    crop_y1 = max(0, int(math.floor(y1 - pad_y)))
    crop_x2 = min(frame_w, int(math.ceil(x2 + pad_x)))
    crop_y2 = min(frame_h, int(math.ceil(y2 + pad_y)))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None
    return (
        frame[crop_y1:crop_y2, crop_x1:crop_x2],
        (crop_x1, crop_y1, crop_x2, crop_y2),
    )


def map_pose_from_crop(
    pose: np.ndarray,
    crop_box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Map normalized MediaPipe crop coordinates to the full camera frame.

    A new array is always returned.  X/Y stay normalized to the full image and
    Z is scaled by the crop-to-frame width ratio, matching BlazePose's
    image-width normalization.
    """

    mapped = np.array(pose, dtype=np.float32, copy=True)
    if mapped.ndim != 2 or mapped.shape[1] < 3:
        return mapped
    if frame_width <= 0 or frame_height <= 0:
        return mapped

    x1, y1, x2, y2 = crop_box
    crop_width = max(0, x2 - x1)
    crop_height = max(0, y2 - y1)
    if crop_width <= 0 or crop_height <= 0:
        return mapped

    mapped[:, 0] = (float(x1) + mapped[:, 0] * crop_width) / frame_width
    mapped[:, 1] = (float(y1) + mapped[:, 1] * crop_height) / frame_height
    mapped[:, 2] *= crop_width / frame_width
    return mapped


def transform_cached_landmarks(
    landmarks: np.ndarray,
    old_box: tuple[int, int, int, int],
    new_box: tuple[int, int, int, int],
    *,
    old_frame_width: int,
    old_frame_height: int,
    new_frame_width: int,
    new_frame_height: int,
) -> np.ndarray:
    """Translate cached landmarks by the tracked bounding-box displacement."""

    transformed = np.array(landmarks, dtype=np.float32, copy=True)
    if transformed.ndim != 2 or transformed.shape[1] < 2:
        return transformed
    if min(
        old_frame_width,
        old_frame_height,
        new_frame_width,
        new_frame_height,
    ) <= 0:
        return transformed

    old_x1, old_y1, old_x2, old_y2 = (float(value) for value in old_box)
    new_x1, new_y1, new_x2, new_y2 = (float(value) for value in new_box)
    if old_x2 <= old_x1 or old_y2 <= old_y1 or new_x2 <= new_x1 or new_y2 <= new_y1:
        return transformed

    old_center_x = (old_x1 + old_x2) * 0.5
    old_center_y = (old_y1 + old_y2) * 0.5
    new_center_x = (new_x1 + new_x2) * 0.5
    new_center_y = (new_y1 + new_y2) * 0.5
    delta_x = new_center_x - old_center_x
    delta_y = new_center_y - old_center_y

    finite_xy = np.isfinite(transformed[:, :2]).all(axis=1)
    old_x_px = transformed[finite_xy, 0] * old_frame_width
    old_y_px = transformed[finite_xy, 1] * old_frame_height
    transformed[finite_xy, 0] = (old_x_px + delta_x) / new_frame_width
    transformed[finite_xy, 1] = (old_y_px + delta_y) / new_frame_height
    return transformed


class PostureAnalyzer:
    """Analyze one camera stream with one MediaPipe state machine."""

    def __init__(
        self,
        estimator: PoseEstimator,
        config: PostureConfig | None = None,
        behavior_classifier: "BehaviorClassifier | None" = None,
    ):
        self.cfg = config or CONFIG.posture
        self.estimator = estimator
        self.behavior_classifier = behavior_classifier
        self.behavior_error: str | None = None
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._last_inference_at = float("-inf")
        self._previous_gray: np.ndarray | None = None
        self._previous_frame_size: tuple[int, int] | None = None

    def close(self) -> None:
        self.estimator.close()

    def process(self, frame: np.ndarray, persons: list[Detection],
                timestamp: float) -> PostureProcessResult:
        persons = [p for p in persons if p.category == "person"]
        self._expire_tracks(timestamp)
        assignments = self._assign_tracks(persons, timestamp)
        frame_h, frame_w = frame.shape[:2]
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sample_interval = 1.0 / max(self.cfg.sample_fps, 0.1)
        if timestamp - self._last_inference_at < sample_interval:
            cached = self._cached_assessments(
                assignments,
                timestamp,
                frame_width=frame_w,
                frame_height=frame_h,
                previous_gray=self._previous_gray,
                current_gray=current_gray,
            )
            self._previous_gray = current_gray
            self._previous_frame_size = (frame_w, frame_h)
            return PostureProcessResult(cached, inference_ran=False)
        self._last_inference_at = timestamp

        eligible_items = [
            (idx, person, track)
            for idx, (person, track) in assignments.items()
            if (person.box[3] - person.box[1]) / max(frame_h, 1)
            >= self.cfg.min_person_height_frac
        ]
        # Pose work grows roughly linearly with the number of crops.  Analyze
        # only the configured number of largest qualifying people; insertion
        # order/index breaks equal-area ties deterministically.
        eligible_items.sort(
            key=lambda item: (
                -max(0, item[1].box[2] - item[1].box[0])
                * max(0, item[1].box[3] - item[1].box[1]),
                item[0],
            )
        )
        max_people = max(0, int(self.cfg.max_poses))
        eligible = {
            idx: (person, track)
            for idx, person, track in eligible_items[:max_people]
        }
        if not eligible:
            self._previous_gray = current_gray
            self._previous_frame_size = (frame_w, frame_h)
            return PostureProcessResult([], inference_ran=True)

        assessments: list[PostureAssessment] = []
        for person_idx, (person, track) in eligible.items():
            crop_box = stabilized_square_crop_box(
                person.box,
                frame_w,
                frame_h,
                previous=track.roi_box,
                margin=self.cfg.crop_margin,
                center_follow=self.cfg.crop_center_follow,
                size_follow=self.cfg.crop_size_follow,
            )
            cropped = crop_from_box(frame, crop_box)
            if cropped is None:
                continue
            crop, crop_box = cropped
            track.roi_box = crop_box
            crop_poses = self.estimator.estimate(crop, int(timestamp * 1000))
            mapped_poses = [
                map_pose_from_crop(pose, crop_box, frame_w, frame_h)
                for pose in crop_poses
            ]
            # A crop normally contains one pose, but matching still prevents a
            # background bystander at its edge from being attached blindly.
            pose = self._match_poses_to_persons(
                mapped_poses,
                {person_idx: person},
                frame_w,
                frame_h,
            ).get(person_idx)
            if pose is None:
                continue
            sample = self._sample_from_pose(pose, person.box, frame_w, frame_h, timestamp)
            if sample is None:
                continue
            track.history.append(sample)
            self._trim_history(track, timestamp)
            self._update_behavior(track, pose, timestamp)
            assessment = self._evaluate_track(
                track,
                person,
                pose,
                timestamp,
                frame_width=frame_w,
                frame_height=frame_h,
            )
            assessment.metrics["pose_age_ms"] = 0.0
            assessment.metrics["optical_flow_points"] = 0.0
            track.latest = assessment
            track.display_landmarks = pose.copy()
            track.display_box = person.box
            track.display_frame_width = frame_w
            track.display_frame_height = frame_h
            assessments.append(assessment)

        self._previous_gray = current_gray
        self._previous_frame_size = (frame_w, frame_h)
        return PostureProcessResult(assessments, inference_ran=True)

    def _expire_tracks(self, now: float) -> None:
        stale = [tid for tid, t in self._tracks.items()
                 if now - t.last_seen > self.cfg.track_ttl_seconds]
        for tid in stale:
            del self._tracks[tid]

    def _assign_tracks(self, persons: list[Detection], now: float
                       ) -> dict[int, tuple[Detection, _Track]]:
        candidates: list[tuple[float, int, int]] = []
        track_ids = list(self._tracks.keys())
        for person_idx, person in enumerate(persons):
            for track_id in track_ids:
                track = self._tracks[track_id]
                iou = _bbox_iou(person.box, track.last_box)
                dist = _box_center_distance(person.box, track.last_box)
                if iou >= self.cfg.track_min_iou or dist <= self.cfg.track_max_center_distance:
                    score = iou + 0.35 * max(0.0, 1.0 - dist)
                    candidates.append((score, person_idx, track_id))

        matched_people: set[int] = set()
        matched_tracks: set[int] = set()
        out: dict[int, tuple[Detection, _Track]] = {}
        for _, person_idx, track_id in sorted(candidates, reverse=True):
            if person_idx in matched_people or track_id in matched_tracks:
                continue
            track = self._tracks[track_id]
            person = persons[person_idx]
            track.last_box = person.box
            track.last_seen = now
            matched_people.add(person_idx)
            matched_tracks.add(track_id)
            out[person_idx] = (person, track)

        for person_idx, person in enumerate(persons):
            if person_idx in matched_people:
                continue
            track = _Track(
                track_id=self._next_track_id,
                last_box=person.box,
                last_seen=now,
            )
            self._next_track_id += 1
            self._tracks[track.track_id] = track
            out[person_idx] = (person, track)
        return out

    def _match_poses_to_persons(
        self,
        poses: list[np.ndarray],
        persons: dict[int, Detection],
        frame_w: int,
        frame_h: int,
    ) -> dict[int, np.ndarray]:
        candidates: list[tuple[float, int, int]] = []
        for pose_idx, pose in enumerate(poses):
            pbox = _pose_bbox(pose, self.cfg.min_landmark_visibility)
            if pbox is None:
                continue
            torso_ids = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
            torso = pose[torso_ids]
            torso_visible = torso[:, 3] >= self.cfg.min_landmark_visibility
            if not np.any(torso_visible):
                continue
            center = np.mean(torso[torso_visible, :2], axis=0)
            for person_idx, person in persons.items():
                bbox = _normalized_box(person.box, frame_w, frame_h)
                pad_x = (bbox[2] - bbox[0]) * 0.12
                pad_y = (bbox[3] - bbox[1]) * 0.08
                center_inside = (
                    bbox[0] - pad_x <= center[0] <= bbox[2] + pad_x
                    and bbox[1] - pad_y <= center[1] <= bbox[3] + pad_y
                )
                visible = pose[:, 3] >= self.cfg.min_landmark_visibility
                xy = pose[visible, :2]
                inside_ratio = 0.0
                if len(xy):
                    inside = (
                        (xy[:, 0] >= bbox[0] - pad_x)
                        & (xy[:, 0] <= bbox[2] + pad_x)
                        & (xy[:, 1] >= bbox[1] - pad_y)
                        & (xy[:, 1] <= bbox[3] + pad_y)
                    )
                    inside_ratio = float(np.mean(inside))
                iou = _normalized_iou(pbox, bbox)
                if center_inside or inside_ratio >= 0.55:
                    score = inside_ratio + 0.45 * iou + (0.2 if center_inside else 0.0)
                    candidates.append((score, person_idx, pose_idx))

        matched_people: set[int] = set()
        matched_poses: set[int] = set()
        out: dict[int, np.ndarray] = {}
        for _, person_idx, pose_idx in sorted(candidates, reverse=True):
            if person_idx in matched_people or pose_idx in matched_poses:
                continue
            matched_people.add(person_idx)
            matched_poses.add(pose_idx)
            out[person_idx] = poses[pose_idx]
        return out

    def _sample_from_pose(
        self,
        pose: np.ndarray,
        bbox: tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
        timestamp: float,
    ) -> PoseSample | None:
        required = pose[[LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]]
        if np.any(required[:, 3] < self.cfg.min_landmark_visibility):
            return None

        left_shoulder, right_shoulder, left_hip, right_hip = required
        shoulder_mid = (left_shoulder[:2] + right_shoulder[:2]) / 2.0
        hip_mid = (left_hip[:2] + right_hip[:2]) / 2.0
        torso_vec = shoulder_mid - hip_mid
        torso_angle = math.degrees(math.atan2(float(torso_vec[0]), -float(torso_vec[1])))

        shoulder_vec = right_shoulder[:2] - left_shoulder[:2]
        shoulder_tilt = math.degrees(
            math.atan2(float(shoulder_vec[1]), float(shoulder_vec[0]))
        )
        shoulder_width = max(float(np.linalg.norm(shoulder_vec)), 1e-5)

        ankle_separation: float | None = None
        ankles = pose[[LEFT_ANKLE, RIGHT_ANKLE]]
        if np.all(ankles[:, 3] >= self.cfg.min_landmark_visibility):
            ankle_separation = abs(float(ankles[0, 0] - ankles[1, 0])) / shoulder_width

        hand_to_mouth: bool | None = None
        mouth = pose[[LEFT_MOUTH, RIGHT_MOUTH]]
        wrists = pose[[15, 16]]
        visible_mouth = mouth[mouth[:, 3] >= self.cfg.min_landmark_visibility, :2]
        visible_wrists = wrists[wrists[:, 3] >= self.cfg.min_landmark_visibility, :2]
        if len(visible_mouth) and len(visible_wrists):
            distances = np.linalg.norm(
                visible_wrists[:, None, :] - visible_mouth[None, :, :],
                axis=2,
            )
            normalized_distance = float(np.min(distances)) / shoulder_width
            hand_to_mouth = normalized_distance <= self.cfg.hand_to_mouth_max_ratio

        visibility_ids = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
                          LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
        pose_confidence = float(np.mean(pose[visibility_ids, 3]))

        return PoseSample(
            timestamp=timestamp,
            torso_angle_deg=float(torso_angle),
            shoulder_tilt_deg=float(shoulder_tilt),
            hip_x=float(hip_mid[0]),
            hip_y=float(hip_mid[1]),
            bbox_width=max((bbox[2] - bbox[0]) / max(frame_w, 1), 1e-5),
            bbox_height=max((bbox[3] - bbox[1]) / max(frame_h, 1), 1e-5),
            ankle_separation=ankle_separation,
            hand_to_mouth=hand_to_mouth,
            pose_confidence=pose_confidence,
        )

    def _trim_history(self, track: _Track, now: float) -> None:
        while track.history and now - track.history[0].timestamp > self.cfg.history_seconds:
            track.history.popleft()

    def _update_behavior(
        self,
        track: _Track,
        pose: np.ndarray,
        timestamp: float,
    ) -> None:
        classifier = self.behavior_classifier
        if classifier is None or not self.cfg.behavior_enabled:
            return

        track.behavior_history.append((float(timestamp), np.asarray(pose, dtype=np.float32).copy()))
        keep_seconds = max(
            2.0,
            classifier.target_window_seconds
            + self.cfg.behavior_max_sample_gap_seconds
            + 0.75,
        )
        cutoff = timestamp - keep_seconds
        while track.behavior_history and track.behavior_history[0][0] < cutoff:
            track.behavior_history.popleft()

        track.behavior_samples_since_inference += 1
        if track.behavior_samples_since_inference < self.cfg.behavior_inference_stride_samples:
            return
        track.behavior_samples_since_inference = 0

        try:
            prediction = classifier.predict_history(list(track.behavior_history))
        except Exception as exc:
            # Keep the existing heuristic detector alive if a runtime model or
            # device error occurs. The manager exposes the startup error, while
            # this field protects against per-frame failures.
            self.behavior_error = str(exc)
            self.behavior_classifier = None
            return
        if prediction is None:
            return

        track.behavior_probability_history.append(dict(prediction.probabilities))
        while len(track.behavior_probability_history) > self.cfg.behavior_smoothing_windows:
            track.behavior_probability_history.popleft()

        labels = list(prediction.probabilities)
        smoothed = {
            label: float(np.mean([
                probabilities.get(label, 0.0)
                for probabilities in track.behavior_probability_history
            ]))
            for label in labels
        }
        label = max(smoothed, key=smoothed.get)
        track.behavior_label = label
        track.behavior_confidence = smoothed[label]
        track.behavior_probabilities = smoothed
        track.behavior_valid_ratio = prediction.valid_ratio
        track.behavior_window_seconds = prediction.window_seconds
        track.behavior_inference_ms = prediction.inference_ms
        track.behavior_prediction_version += 1

    def _merge_behavior_assessment(
        self,
        track: _Track,
        assessment: PostureAssessment,
        timestamp: float,
    ) -> PostureAssessment:
        if not track.behavior_probabilities:
            return assessment

        probabilities = dict(track.behavior_probabilities)
        fall_probability = float(probabilities.get("fall_down", 0.0))
        lying_probability = float(probabilities.get("lying_down", 0.0))
        fall_detected = fall_probability >= self.cfg.behavior_fall_threshold
        lying_detected = lying_probability >= self.cfg.behavior_lying_threshold

        metrics = dict(assessment.metrics)
        metrics["behavior_valid_ratio"] = round(track.behavior_valid_ratio, 4)
        metrics["behavior_window_seconds"] = round(track.behavior_window_seconds, 4)
        metrics["behavior_inference_ms"] = round(track.behavior_inference_ms, 3)
        for label, probability in probabilities.items():
            metrics[f"behavior_probability_{label}"] = round(float(probability), 4)

        signals = list(assessment.signals)
        risk = assessment.risk_score
        severity = assessment.severity
        status = assessment.status
        confirmed = assessment.confirmed

        if fall_detected:
            if "ml_fall_down" not in signals:
                signals.append("ml_fall_down")
            risk = max(risk, fall_probability, self.cfg.danger_score)
            severity = "DANGER"
            status = "high_risk"
        elif lying_detected:
            if "ml_lying_down" not in signals:
                signals.append("ml_lying_down")
            risk = max(risk, self.cfg.warning_score, 0.90 * lying_probability)
            if severity != "DANGER":
                severity = "WARNING"
                status = "verification_required"

        new_prediction = track.behavior_prediction_version > track.behavior_streak_version
        if new_prediction:
            track.behavior_streak_version = track.behavior_prediction_version
            if fall_detected:
                track.behavior_fall_streak += 1
                track.behavior_lying_streak = 0
            elif lying_detected:
                track.behavior_lying_streak += 1
                track.behavior_fall_streak = 0
            else:
                track.behavior_fall_streak = 0
                track.behavior_lying_streak = 0

        enough_fall = (
            fall_detected
            and track.behavior_fall_streak
            >= self.cfg.behavior_fall_consecutive_windows
        )
        enough_lying = (
            lying_detected
            and track.behavior_lying_streak
            >= self.cfg.behavior_lying_consecutive_windows
        )

        # Escalate sustained lying after temporal confirmation.
        if enough_lying:
            if "ml_lying_down_confirmed" not in signals:
                signals.append("ml_lying_down_confirmed")
            risk = max(risk, lying_probability, self.cfg.danger_score)
            severity = "DANGER"
            status = "high_risk"

        if (
            new_prediction
            and self.cfg.behavior_alerts_enabled
            and (enough_fall or enough_lying)
            and timestamp - track.behavior_last_alert_time >= self.cfg.cooldown_seconds
        ):
            confirmed = True
            track.behavior_last_alert_time = timestamp

        return replace(
            assessment,
            risk_score=_clip01(risk),
            severity=severity,
            status=status,
            signals=signals,
            metrics=metrics,
            confirmed=confirmed,
            behavior_label=track.behavior_label,
            behavior_confidence=track.behavior_confidence,
            behavior_probabilities=probabilities,
            behavior_valid_ratio=track.behavior_valid_ratio,
            behavior_window_seconds=track.behavior_window_seconds,
            behavior_inference_ms=track.behavior_inference_ms,
        )

    def _evaluate_track(
        self,
        track: _Track,
        person: Detection,
        pose: np.ndarray,
        timestamp: float,
        *,
        frame_width: int,
        frame_height: int,
    ) -> PostureAssessment:
        history = list(track.history)
        duration = history[-1].timestamp - history[0].timestamp if len(history) > 1 else 0.0
        pose_conf = float(np.mean([s.pose_confidence for s in history])) if history else 0.0

        if len(history) < self.cfg.min_samples or duration < self.cfg.min_history_seconds:
            track.alert_streak = 0
            assessment = PostureAssessment(
                track_id=track.track_id,
                person=person,
                risk_score=0.0,
                severity="OK",
                status="collecting_history",
                signals=[],
                metrics={"samples": float(len(history))},
                frame_timestamp=timestamp,
                pose_confidence=pose_conf,
                history_seconds=duration,
                landmarks=pose.copy(),
                frame_width=frame_width,
                frame_height=frame_height,
            )
            return self._merge_behavior_assessment(track, assessment, timestamp)

        times = np.asarray([s.timestamp for s in history], dtype=np.float64)
        times -= times[0]
        torso = _smooth(np.asarray([s.torso_angle_deg for s in history], dtype=np.float64))
        shoulder = _smooth(np.asarray([s.shoulder_tilt_deg for s in history], dtype=np.float64))
        hip_x = _smooth(np.asarray([s.hip_x for s in history], dtype=np.float64))
        hip_y_raw = np.asarray([s.hip_y for s in history], dtype=np.float64)
        hip_y = _smooth(hip_y_raw)
        box_w = float(np.median([s.bbox_width for s in history]))
        box_h = float(np.median([s.bbox_height for s in history]))

        torso_sway_deg = float(np.std(torso))
        shoulder_tilt_std_deg = float(np.std(shoulder))

        # Remove the person's overall walking direction.  The remaining x
        # residual measures side-to-side motion in units of person-box width.
        if len(times) >= 2 and times[-1] > 0:
            slope, intercept = np.polyfit(times, hip_x, 1)
            lateral_residual = hip_x - (slope * times + intercept)
        else:
            lateral_residual = hip_x - np.mean(hip_x)
        trajectory_sway = float(np.std(lateral_residual) / max(box_w, 1e-5))

        ankle_values = np.asarray(
            [s.ankle_separation for s in history if s.ankle_separation is not None],
            dtype=np.float64,
        )
        if len(ankle_values) >= max(4, self.cfg.min_samples // 2):
            ankle_values = _smooth(ankle_values)
            step_variability = float(np.std(np.diff(ankle_values)))
        else:
            step_variability = 0.0

        sudden_drop_speed = 0.0
        dt = np.diff(times)
        if len(dt) and np.any(dt > 1e-4):
            # Keep the short fall impulse before smoothing; temporal
            # confirmation and the horizontal-torso requirement suppress noise.
            dy = np.diff(hip_y_raw)
            speeds = dy[dt > 1e-4] / dt[dt > 1e-4]
            if len(speeds):
                sudden_drop_speed = max(0.0, float(np.max(speeds)) / max(box_h, 1e-5))

        torso_component = _component(
            torso_sway_deg,
            self.cfg.torso_sway_start_deg,
            self.cfg.torso_sway_full_deg,
        )
        trajectory_component = _component(
            trajectory_sway,
            self.cfg.trajectory_sway_start,
            self.cfg.trajectory_sway_full,
        )
        step_component = _component(
            step_variability,
            self.cfg.step_variability_start,
            self.cfg.step_variability_full,
        )
        shoulder_component = _component(
            shoulder_tilt_std_deg,
            self.cfg.shoulder_tilt_start_deg,
            self.cfg.shoulder_tilt_full_deg,
        )
        drop_component = _component(
            sudden_drop_speed,
            self.cfg.sudden_drop_start,
            self.cfg.sudden_drop_full,
        )
        recent_count = max(3, min(len(torso), int(round(self.cfg.sample_fps * 1.2))))
        recent_torso = torso[-recent_count:]
        horizontal_fraction = float(np.mean(
            np.abs(recent_torso) >= self.cfg.fall_torso_angle_deg
        )) if len(recent_torso) else 0.0
        possible_fall = (
            drop_component >= self.cfg.fall_drop_component
            and horizontal_fraction >= self.cfg.fall_horizontal_fraction
        )

        gesture_start = timestamp - self.cfg.hand_to_mouth_window_seconds
        gesture_values = [
            sample.hand_to_mouth
            for sample in history
            if sample.timestamp >= gesture_start and sample.hand_to_mouth is not None
        ]
        hand_to_mouth_fraction = (
            float(np.mean(gesture_values)) if gesture_values else 0.0
        )
        hand_to_mouth_pattern = (
            self.cfg.hand_to_mouth_enabled
            and len(gesture_values) >= self.cfg.hand_to_mouth_min_samples
            and hand_to_mouth_fraction >= self.cfg.hand_to_mouth_fraction
        )

        risk = (
            0.34 * torso_component
            + 0.31 * trajectory_component
            + 0.19 * step_component
            + 0.10 * shoulder_component
            + 0.06 * drop_component
        )
        # Two independent abnormal cues are more meaningful than one noisy
        # landmark stream.  This small synergy bonus is still bounded and is
        # only applied after temporal smoothing.
        strong_components = sum(
            component >= 0.45
            for component in (
                torso_component, trajectory_component, step_component,
                shoulder_component, drop_component,
            )
        )
        if strong_components >= 2:
            risk += 0.12
        if strong_components >= 3:
            risk += 0.08
        if drop_component >= 0.8:
            risk = max(risk, self.cfg.danger_score)
        if possible_fall:
            risk = max(risk, min(1.0, self.cfg.danger_score + 0.08))
        # Downweight poor skeletons instead of treating detector jitter as a
        # behavioural signal.
        confidence_factor = _clip01((pose_conf - 0.45) / 0.40)
        risk *= 0.55 + 0.45 * confidence_factor
        if hand_to_mouth_pattern:
            # This is an independent behaviour cue, not evidence of impaired
            # coordination. The score is only lifted to verification level.
            risk = max(risk, min(1.0, self.cfg.warning_score + 0.05))
        risk = _clip01(risk)

        signals: list[str] = []
        if torso_component >= 0.30:
            signals.append("repeated_body_sway")
        if trajectory_component >= 0.45:
            signals.append("unstable_trajectory")
        if step_component >= 0.45:
            signals.append("irregular_step_pattern")
        if shoulder_component >= 0.55:
            signals.append("upper_body_instability")
        if drop_component >= 0.65:
            signals.append("sudden_balance_loss")
        if possible_fall:
            signals.append("possible_fall")
        if hand_to_mouth_pattern:
            signals.append("hand_to_mouth_pattern")

        if risk >= self.cfg.danger_score:
            severity, status = "DANGER", "high_risk"
        elif risk >= self.cfg.warning_score:
            severity, status = "WARNING", "verification_required"
        elif risk >= self.cfg.observation_score:
            severity, status = "OK", "observation"
        else:
            severity, status = "OK", "normal"

        confirmed = False
        if possible_fall:
            # A fall is a distinct critical event and must not be hidden by the
            # cooldown of an earlier generic coordination warning.
            track.fall_streak += 1
            track.gesture_streak = 0
            if (
                track.fall_streak >= 1
                and timestamp - track.last_fall_alert_time >= self.cfg.cooldown_seconds
            ):
                confirmed = True
                track.last_fall_alert_time = timestamp
            track.alert_streak = 0
        elif hand_to_mouth_pattern:
            track.fall_streak = 0
            track.alert_streak = 0
            track.gesture_streak += 1
            if (
                track.gesture_streak >= self.cfg.hand_to_mouth_consecutive_windows
                and timestamp - track.last_gesture_alert_time >= self.cfg.cooldown_seconds
            ):
                confirmed = True
                track.last_gesture_alert_time = timestamp
        else:
            track.fall_streak = 0
            track.gesture_streak = 0
            if risk >= self.cfg.warning_score:
                track.alert_streak += 1
                if (
                    track.alert_streak >= self.cfg.consecutive_windows_required
                    and timestamp - track.last_alert_time >= self.cfg.cooldown_seconds
                ):
                    confirmed = True
                    track.last_alert_time = timestamp
            else:
                track.alert_streak = 0

        metrics = {
            "torso_sway_deg": round(torso_sway_deg, 3),
            "trajectory_sway_ratio": round(trajectory_sway, 4),
            "step_variability": round(step_variability, 4),
            "shoulder_tilt_std_deg": round(shoulder_tilt_std_deg, 3),
            "sudden_drop_box_per_sec": round(sudden_drop_speed, 4),
            "horizontal_torso_fraction": round(horizontal_fraction, 4),
            "hand_to_mouth_fraction": round(hand_to_mouth_fraction, 4),
            "samples": float(len(history)),
        }
        assessment = PostureAssessment(
            track_id=track.track_id,
            person=person,
            risk_score=risk,
            severity=severity,
            status=status,
            signals=signals,
            metrics=metrics,
            frame_timestamp=timestamp,
            pose_confidence=pose_conf,
            history_seconds=duration,
            confirmed=confirmed,
            landmarks=pose.copy(),
            frame_width=frame_width,
            frame_height=frame_height,
        )
        return self._merge_behavior_assessment(track, assessment, timestamp)

    def _cached_assessments(
        self,
        assignments: dict[int, tuple[Detection, _Track]],
        now: float,
        *,
        frame_width: int,
        frame_height: int,
        previous_gray: np.ndarray | None = None,
        current_gray: np.ndarray | None = None,
    ) -> list[PostureAssessment]:
        out: list[PostureAssessment] = []
        for person, track in assignments.values():
            latest = track.latest
            if latest is None or now - latest.frame_timestamp > self.cfg.cached_result_ttl_seconds:
                continue
            source_landmarks = track.display_landmarks if track.display_landmarks is not None else latest.landmarks
            source_box = track.display_box or latest.person.box
            old_frame_width = track.display_frame_width or latest.frame_width or frame_width
            old_frame_height = track.display_frame_height or latest.frame_height or frame_height
            landmarks = (
                transform_cached_landmarks(
                    source_landmarks,
                    source_box,
                    person.box,
                    old_frame_width=old_frame_width,
                    old_frame_height=old_frame_height,
                    new_frame_width=frame_width,
                    new_frame_height=frame_height,
                )
                if source_landmarks is not None
                else None
            )
            flow_points = 0
            if (
                landmarks is not None
                and source_landmarks is not None
                and self.cfg.optical_flow_enabled
                and previous_gray is not None
                and current_gray is not None
                and old_frame_width == frame_width
                and old_frame_height == frame_height
            ):
                landmarks, flow_points = track_pose_optical_flow(
                    previous_gray,
                    current_gray,
                    source_landmarks,
                    landmarks,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    bbox=person.box,
                    min_visibility=self.cfg.min_landmark_visibility,
                    win_size=self.cfg.optical_flow_win_size,
                    max_level=self.cfg.optical_flow_max_level,
                    fb_threshold_px=self.cfg.optical_flow_fb_threshold_px,
                    max_jump_frac=self.cfg.optical_flow_max_jump_frac,
                )
            if landmarks is not None:
                track.display_landmarks = landmarks.copy()
                track.display_box = person.box
                track.display_frame_width = frame_width
                track.display_frame_height = frame_height
            metrics = dict(latest.metrics)
            metrics["pose_age_ms"] = max(0.0, (now - latest.frame_timestamp) * 1000.0)
            metrics["optical_flow_points"] = float(flow_points)
            # Keep risk/status from real MediaPipe samples; optical flow is only
            # a low-cost display tracker and never creates behavioural alerts.
            out.append(replace(
                latest,
                person=person,
                frame_timestamp=now,
                confirmed=False,
                landmarks=landmarks,
                metrics=metrics,
                frame_width=frame_width,
                frame_height=frame_height,
            ))
        return out


class PostureManager:
    """Lazy per-camera analyzer registry used by the FastAPI application."""

    def __init__(
        self,
        config: PostureConfig | None = None,
        estimator_factory: Callable[[PostureConfig], PoseEstimator] | None = None,
        behavior_classifier: "BehaviorClassifier | None" = None,
    ):
        self.cfg = config or CONFIG.posture
        self._factory = estimator_factory or MediaPipePoseEstimator
        self._analyzers: dict[str, PostureAnalyzer] = {}
        self._last_used: dict[str, float] = {}
        self.available, self.unavailable_reason = self._check_available()
        self.behavior_classifier = behavior_classifier
        self.behavior_available = behavior_classifier is not None
        self.behavior_unavailable_reason: str | None = None
        # Custom estimator factories are predominantly used by deterministic
        # tests; do not load a real checkpoint behind their back.
        if (
            self.behavior_classifier is None
            and self.cfg.behavior_enabled
            and self._factory is MediaPipePoseEstimator
        ):
            try:
                from backend.behavior_classifier import BehaviorClassifier
                self.behavior_classifier = BehaviorClassifier(
                    self.cfg.behavior_model_path,
                    device=self.cfg.behavior_device,
                    feature_fps=self.cfg.behavior_feature_fps,
                    min_valid_ratio=self.cfg.behavior_min_valid_ratio,
                    min_window_coverage=self.cfg.behavior_min_window_coverage,
                    max_sample_gap_seconds=self.cfg.behavior_max_sample_gap_seconds,
                )
                self.behavior_available = True
            except Exception as exc:
                self.behavior_available = False
                self.behavior_unavailable_reason = str(exc)

    def _check_available(self) -> tuple[bool, str | None]:
        if not self.cfg.enabled:
            return False, "disabled by POSTURE_ENABLED"
        if importlib.util.find_spec("mediapipe") is None and self._factory is MediaPipePoseEstimator:
            return False, "mediapipe package is not installed"
        if self._factory is MediaPipePoseEstimator and not os.path.exists(self.cfg.model_path):
            return False, f"model file not found: {self.cfg.model_path}"
        return True, None

    def process(self, camera_id: str, frame: np.ndarray,
                persons: list[Detection], timestamp: float) -> PostureProcessResult:
        if not self.available:
            return PostureProcessResult([], inference_ran=False)
        analyzer = self._analyzers.get(camera_id)
        if analyzer is None:
            self._evict_if_needed()
            try:
                analyzer = PostureAnalyzer(
                    self._factory(self.cfg),
                    self.cfg,
                    behavior_classifier=self.behavior_classifier,
                )
            except Exception as exc:
                self.available = False
                self.unavailable_reason = str(exc)
                return PostureProcessResult([], inference_ran=False)
            self._analyzers[camera_id] = analyzer
        self._last_used[camera_id] = time.monotonic()
        return analyzer.process(frame, persons, timestamp)

    def _evict_if_needed(self) -> None:
        if len(self._analyzers) < self.cfg.max_camera_instances:
            return
        oldest = min(self._last_used, key=self._last_used.get)
        self._analyzers.pop(oldest).close()
        self._last_used.pop(oldest, None)

    def reset_camera(self, camera_id: str) -> None:
        analyzer = self._analyzers.pop(camera_id, None)
        self._last_used.pop(camera_id, None)
        if analyzer is not None:
            analyzer.close()

    def close(self) -> None:
        for analyzer in self._analyzers.values():
            analyzer.close()
        self._analyzers.clear()
        self._last_used.clear()


@dataclass(frozen=True)
class _PostureWorkerJob:
    camera_id: str
    frame: np.ndarray
    persons: tuple[Detection, ...]
    timestamp: float


class PostureWorker:
    """Run posture processing outside the ingest request with latest-wins.

    There is exactly one global pending slot in addition to the job currently
    being processed.  Submitting another frame replaces that pending job, so a
    slow MediaPipe call cannot create an ever-growing, increasingly stale
    queue.  Completed snapshots are retained per camera in a bounded map.

    The worker deliberately is not wired into FastAPI here.  Callers can later
    integrate it using ``submit_latest`` in ingest and
    ``get_latest_result``/``get_latest`` when building a response.
    """

    def __init__(
        self,
        manager: PostureManager,
        *,
        max_result_cameras: int | None = None,
        close_manager: bool = False,
        autostart: bool = True,
        thread_name: str = "posture-worker",
    ):
        self.manager = manager
        configured_limit = max(1, int(manager.cfg.max_camera_instances))
        self.max_result_cameras = max(
            1,
            int(max_result_cameras or configured_limit),
        )
        self.close_manager = close_manager
        self.thread_name = thread_name

        self._condition = threading.Condition()
        self._manager_close_lock = threading.Lock()
        self._pending: _PostureWorkerJob | None = None
        self._results: OrderedDict[str, PostureWorkerSnapshot] = OrderedDict()
        self._active = False
        self._closed = False
        self._manager_closed = False
        self._disabled_cameras: set[str] = set()
        self._thread: threading.Thread | None = None
        if autostart:
            self.start()

    def start(self) -> None:
        """Start the daemon thread; repeated calls are harmless."""

        with self._condition:
            if self._closed:
                raise RuntimeError("PostureWorker is closed")
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run,
                name=self.thread_name,
                daemon=True,
            )
            self._thread.start()

    def submit_latest(
        self,
        camera_id: str,
        frame: np.ndarray,
        persons: list[Detection],
        timestamp: float,
    ) -> bool:
        """Submit a safe private snapshot, replacing any pending older job.

        Returns ``False`` only after the worker has been closed.
        """

        if not camera_id:
            raise ValueError("camera_id must not be empty")
        frame_copy = np.array(frame, copy=True)
        people_copy = tuple(replace(person) for person in persons)
        job = _PostureWorkerJob(
            camera_id=camera_id,
            frame=frame_copy,
            persons=people_copy,
            timestamp=float(timestamp),
        )
        with self._condition:
            if self._closed or camera_id in self._disabled_cameras:
                return False
            self._pending = job
            self._condition.notify()
            return True

    def set_camera_enabled(self, camera_id: str, enabled: bool) -> None:
        """Enable/disable submissions and discard stale results dynamically."""
        with self._condition:
            if enabled:
                self._disabled_cameras.discard(camera_id)
            else:
                self._disabled_cameras.add(camera_id)
                if self._pending is not None and self._pending.camera_id == camera_id:
                    self._pending = None
                self._results.pop(camera_id, None)
            self._condition.notify_all()

    @property
    def pending_count(self) -> int:
        """Number of jobs waiting behind the active call (always zero or one)."""

        with self._condition:
            return int(self._pending is not None)

    @property
    def busy(self) -> bool:
        with self._condition:
            return self._active or self._pending is not None

    def get_latest(self, camera_id: str) -> PostureWorkerSnapshot | None:
        with self._condition:
            return self._results.get(camera_id)

    def get_latest_result(self, camera_id: str) -> PostureProcessResult | None:
        """Convenience accessor matching the intended ingest integration."""

        snapshot = self.get_latest(camera_id)
        return snapshot.result if snapshot is not None else None

    def wait_for_result(
        self,
        camera_id: str,
        *,
        after_timestamp: float | None = None,
        timeout: float = 2.0,
    ) -> PostureWorkerSnapshot | None:
        """Wait for a camera result newer than ``after_timestamp`` in tests/tools."""

        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while True:
                snapshot = self._results.get(camera_id)
                if snapshot is not None and (
                    after_timestamp is None
                    or snapshot.frame_timestamp > after_timestamp
                ):
                    return snapshot
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                if self._closed and not self._active and self._pending is None:
                    return None
                self._condition.wait(remaining)

    def close(self, *, wait: bool = True, timeout: float | None = 5.0) -> None:
        """Drop the pending job and stop after any active native call finishes."""

        with self._condition:
            if self._closed:
                thread = self._thread
            else:
                self._closed = True
                self._pending = None
                thread = self._thread
                self._condition.notify_all()
        if (
            wait
            and thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=timeout)
        if self.close_manager and (thread is None or not thread.is_alive()):
            self._close_manager_once()

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._closed:
                        self._condition.wait()
                    if self._closed:
                        return
                    job = self._pending
                    self._pending = None
                    self._active = True

                assert job is not None
                result: PostureProcessResult | None = None
                error: str | None = None
                try:
                    result = self.manager.process(
                        job.camera_id,
                        job.frame,
                        list(job.persons),
                        job.timestamp,
                    )
                except Exception as exc:  # keep the long-lived worker alive
                    error = f"{type(exc).__name__}: {exc}"

                snapshot = PostureWorkerSnapshot(
                    camera_id=job.camera_id,
                    frame_timestamp=job.timestamp,
                    completed_at=time.time(),
                    result=result,
                    error=error,
                )
                with self._condition:
                    if job.camera_id not in self._disabled_cameras:
                        self._results[job.camera_id] = snapshot
                        self._results.move_to_end(job.camera_id)
                        while len(self._results) > self.max_result_cameras:
                            self._results.popitem(last=False)
                    self._active = False
                    self._condition.notify_all()
        finally:
            if self.close_manager:
                self._close_manager_once()

    def _close_manager_once(self) -> None:
        with self._manager_close_lock:
            if self._manager_closed:
                return
            self.manager.close()
            self._manager_closed = True
