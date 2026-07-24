"""Metric camera calibration from four ArUco markers.

New calibrations store a homography from normalized image coordinates
``[u, v] in [0, 1]`` to metres.  This makes a calibration independent of
resolution while still rejecting a different crop/aspect ratio.

Older JSON records used a pixel-to-metre homography and did not persist the
source dimensions.  They remain loadable for audit/backward-compatible unit
use, but ``is_compatible`` deliberately returns ``False`` so live processing
can require a fresh normalized calibration.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import cv2
import numpy as np

from backend.marker_detector import MarkerDetection


DEFAULT_ASPECT_TOLERANCE = 0.01
MIN_COVERAGE_RATIO = 0.02
MIN_EDGE_RATIO = 0.05
MIN_INTERIOR_ANGLE_DEG = 15.0
MAX_INTERIOR_ANGLE_DEG = 165.0
MAX_HOMOGRAPHY_CONDITION = 1.0e10
MIN_MARKER_AREA_PX = 16.0


class CalibrationError(ValueError):
    """Base class for actionable calibration failures."""


class CalibrationGeometryError(CalibrationError):
    """The marker arrangement cannot produce a trustworthy homography."""


class CalibrationCompatibilityError(CalibrationError):
    """A calibrated homography is incompatible with a requested frame."""


def _legacy_quality() -> dict[str, Any]:
    return {
        "valid": False,
        "status": "invalid",
        "score": 0.0,
        "score_percent": 0.0,
        "coverage_ratio": 0.0,
        "quadrilateral_area_ratio": 0.0,
        "min_edge_ratio": 0.0,
        "max_edge_ratio": 0.0,
        "smallest_marker_area_px": 0.0,
        "min_angle_deg": 0.0,
        "max_angle_deg": 0.0,
        "reprojection_error_normalized": None,
        "reprojection_error_m": None,
        "condition_number": None,
        "warnings": [
            "Legacy calibration has no source frame geometry; recalibration is required."
        ],
        "messages": [
            "Legacy calibration has no source frame geometry; recalibration is required."
        ],
    }


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise CalibrationGeometryError(f"{name} must be a positive finite number.")
    return value


def _frame_dimensions(width: int, height: int) -> tuple[int, int]:
    if isinstance(width, bool) or isinstance(height, bool):
        raise CalibrationGeometryError("Frame width and height must be positive integers.")
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise CalibrationGeometryError("Frame width and height must be positive integers.")
    return width, height


def _validate_marker_ids(marker_ids: list[int]) -> list[int]:
    if len(marker_ids) != 4:
        raise CalibrationGeometryError(
            "Need exactly 4 marker IDs in TL, TR, BR, BL order."
        )
    parsed = [int(marker_id) for marker_id in marker_ids]
    if len(set(parsed)) != 4:
        raise CalibrationGeometryError("Marker IDs must be unique.")
    invalid = [marker_id for marker_id in parsed if marker_id < 0 or marker_id >= 50]
    if invalid:
        raise CalibrationGeometryError(
            f"Marker IDs outside DICT_4X4_50 range 0..49: {invalid}"
        )
    return parsed


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _interior_angles(points: np.ndarray) -> list[float]:
    angles: list[float] = []
    for index in range(4):
        previous = points[(index - 1) % 4] - points[index]
        following = points[(index + 1) % 4] - points[index]
        denominator = float(np.linalg.norm(previous) * np.linalg.norm(following))
        if denominator <= 1e-12:
            angles.append(0.0)
            continue
        cosine = float(np.dot(previous, following) / denominator)
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    return angles


def _geometry_quality(
    normalized_points: np.ndarray,
    homography: np.ndarray,
    world_points: np.ndarray,
    smallest_marker_area_px: float,
) -> dict[str, Any]:
    edges = [
        float(np.linalg.norm(normalized_points[(i + 1) % 4] - normalized_points[i]))
        for i in range(4)
    ]
    angles = _interior_angles(normalized_points)
    coverage = abs(_polygon_area(normalized_points))

    projected_world = cv2.perspectiveTransform(
        normalized_points.astype(np.float64).reshape(1, -1, 2),
        homography.astype(np.float64),
    ).reshape(-1, 2)
    reprojection_m = float(
        np.sqrt(np.mean(np.sum((projected_world - world_points) ** 2, axis=1)))
    )
    inverse = np.linalg.inv(homography)
    projected_normalized = cv2.perspectiveTransform(
        world_points.astype(np.float64).reshape(1, -1, 2),
        inverse.astype(np.float64),
    ).reshape(-1, 2)
    reprojection_normalized = float(
        np.sqrt(
            np.mean(np.sum((projected_normalized - normalized_points) ** 2, axis=1))
        )
    )

    normalized_h = homography / homography[2, 2]
    condition = float(np.linalg.cond(normalized_h))
    min_edge, max_edge = min(edges), max(edges)
    min_angle, max_angle = min(angles), max(angles)

    warnings: list[str] = []
    if coverage < 0.08:
        warnings.append(
            "Reference quadrilateral occupies less than 8% of the frame; "
            "move the camera closer or spread markers farther apart."
        )
    if min_edge < 0.12:
        warnings.append(
            "At least one reference edge is short in the image; metric precision may be low."
        )
    if min_angle < 35.0 or max_angle > 145.0:
        warnings.append(
            "Strong perspective/skew detected; use a less oblique camera angle if possible."
        )
    if condition > 1.0e5:
        warnings.append("Homography is ill-conditioned; measurements may be unstable.")

    coverage_score = min(1.0, coverage / 0.25)
    edge_score = min(1.0, min_edge / 0.20)
    angle_deviation = max(abs(angle - 90.0) for angle in angles)
    angle_score = max(0.0, 1.0 - angle_deviation / 90.0)
    condition_score = max(
        0.0,
        min(1.0, 1.0 - max(0.0, math.log10(max(condition, 1.0)) - 3.0) / 7.0),
    )
    score = (
        0.35 * coverage_score
        + 0.25 * edge_score
        + 0.25 * angle_score
        + 0.15 * condition_score
    )
    status = "good" if score >= 0.75 and not warnings else "warning"
    return {
        "valid": True,
        "status": status,
        "score": round(float(score), 4),
        "score_percent": round(float(score) * 100.0, 1),
        "coverage_ratio": round(coverage, 6),
        "quadrilateral_area_ratio": round(coverage, 6),
        "min_edge_ratio": round(min_edge, 6),
        "max_edge_ratio": round(max_edge, 6),
        "smallest_marker_area_px": round(float(smallest_marker_area_px), 3),
        "min_angle_deg": round(min_angle, 3),
        "max_angle_deg": round(max_angle, 3),
        "reprojection_error_normalized": round(reprojection_normalized, 10),
        "reprojection_error_m": round(reprojection_m, 10),
        "condition_number": round(condition, 3),
        "warnings": warnings,
        "messages": list(warnings),
    }


@dataclass
class Calibration:
    """Homography from normalized image coordinates to a metric plane.

    ``marker_centers`` are normalized ``[u, v]`` pairs ordered exactly like
    ``marker_ids``: TL, TR, BR, BL. ``marker_centers_px`` are persisted only
    for diagnostics against the original source frame.
    """

    camera_id: str
    marker_ids: list[int]
    width_m: float
    height_m: float
    homography: list[list[float]]
    source_frame_width: int = 0
    source_frame_height: int = 0
    source_aspect_ratio: float = 0.0
    marker_centers: list[list[float]] = field(default_factory=list)
    marker_centers_px: list[list[float]] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=_legacy_quality)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.homography, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise CalibrationGeometryError("Homography must be a finite 3x3 matrix.")
        if abs(float(np.linalg.det(matrix))) <= 1e-15:
            raise CalibrationGeometryError("Homography is singular.")
        self.width_m = _positive_finite(self.width_m, "width_m")
        self.height_m = _positive_finite(self.height_m, "height_m")
        self.marker_ids = [int(marker_id) for marker_id in self.marker_ids]
        self.source_frame_width = int(self.source_frame_width or 0)
        self.source_frame_height = int(self.source_frame_height or 0)
        if self.source_frame_width > 0 and self.source_frame_height > 0:
            calculated = self.source_frame_width / self.source_frame_height
            # Dimensions are canonical; never trust a stale/rounded persisted
            # ratio over the actual source geometry.
            self.source_aspect_ratio = calculated
        else:
            self.source_aspect_ratio = 0.0

    @property
    def source_frame_aspect(self) -> float:
        """Deprecated compatibility alias for pre-spec callers."""
        return self.source_aspect_ratio

    @property
    def normalized(self) -> bool:
        """Whether this record safely maps normalized frame coordinates."""
        return (
            self.source_frame_width > 0
            and self.source_frame_height > 0
            and self.source_aspect_ratio > 0.0
        )

    def compatibility_warning(
        self,
        frame_width: int,
        frame_height: int,
        tolerance: float = DEFAULT_ASPECT_TOLERANCE,
    ) -> str | None:
        if not self.normalized:
            return (
                "Legacy calibration has no source frame dimensions and cannot "
                "be applied safely; recalibrate this camera."
            )
        try:
            width, height = _frame_dimensions(frame_width, frame_height)
        except CalibrationGeometryError as exc:
            return str(exc)
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            return "Aspect-ratio tolerance must be a non-negative finite number."
        current_aspect = width / height
        relative_delta = abs(current_aspect - self.source_aspect_ratio) / self.source_aspect_ratio
        if relative_delta > tolerance + 1e-12:
            return (
                "Calibration aspect mismatch: calibrated at "
                f"{self.source_frame_width}x{self.source_frame_height} "
                f"({self.source_aspect_ratio:.6f}), current frame is "
                f"{width}x{height} ({current_aspect:.6f}); "
                f"difference {relative_delta * 100.0:.2f}% exceeds "
                f"{tolerance * 100.0:.2f}%. Recalibrate for this crop/aspect."
            )
        return None

    def is_compatible(
        self,
        frame_width: int,
        frame_height: int,
        tolerance: float = DEFAULT_ASPECT_TOLERANCE,
    ) -> bool:
        return self.compatibility_warning(frame_width, frame_height, tolerance) is None

    def _resolve_frame(
        self,
        frame_width: int | None,
        frame_height: int | None,
    ) -> tuple[int, int] | None:
        # A legacy record can still serve old direct unit tests when no target
        # dimensions are supplied. Live code should call is_compatible first.
        if not self.normalized:
            if frame_width is None and frame_height is None:
                return None
            raise CalibrationCompatibilityError(
                self.compatibility_warning(
                    int(frame_width or 0), int(frame_height or 0)
                )
                or "Legacy calibration requires recalibration."
            )
        if (frame_width is None) != (frame_height is None):
            raise CalibrationCompatibilityError(
                "frame_width and frame_height must be supplied together."
            )
        width = self.source_frame_width if frame_width is None else int(frame_width)
        height = self.source_frame_height if frame_height is None else int(frame_height)
        warning = self.compatibility_warning(width, height)
        if warning:
            raise CalibrationCompatibilityError(warning)
        return width, height

    def project(
        self,
        px: float,
        py: float,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> tuple[float, float]:
        """Project a pixel point to metres, rejecting a different aspect."""
        frame = self._resolve_frame(frame_width, frame_height)
        source = np.array(
            [float(px), float(py), 1.0]
            if frame is None
            else [float(px) / frame[0], float(py) / frame[1], 1.0],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(source)):
            raise CalibrationGeometryError("Pixel coordinates must be finite.")
        vector = np.asarray(self.homography, dtype=np.float64) @ source
        if abs(float(vector[2])) < 1e-12:
            raise CalibrationGeometryError("Point projects to infinity.")
        return (
            float(vector[0] / vector[2]),
            float(vector[1] / vector[2]),
        )

    def unproject(
        self,
        x_m: float,
        y_m: float,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> tuple[float, float]:
        """Project a metric-plane point back to pixels."""
        frame = self._resolve_frame(frame_width, frame_height)
        metric = np.array([float(x_m), float(y_m), 1.0], dtype=np.float64)
        if not np.all(np.isfinite(metric)):
            raise CalibrationGeometryError("Metric coordinates must be finite.")
        try:
            inverse = np.linalg.inv(np.asarray(self.homography, dtype=np.float64))
        except np.linalg.LinAlgError as exc:  # guarded by __post_init__
            raise CalibrationGeometryError("Homography is singular.") from exc
        vector = inverse @ metric
        if abs(float(vector[2])) < 1e-12:
            raise CalibrationGeometryError("Point unprojects to infinity.")
        x, y = float(vector[0] / vector[2]), float(vector[1] / vector[2])
        if frame is not None:
            x *= frame[0]
            y *= frame[1]
        return x, y

    def distance_to_boundary_m(
        self,
        px: float,
        py: float,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> float:
        """Signed distance to reference-rectangle boundary in metres."""
        x, y = self.project(px, py, frame_width, frame_height)
        dx = max(-x, x - self.width_m, 0.0)
        dy = max(-y, y - self.height_m, 0.0)
        if dx > 0.0 or dy > 0.0:
            return float(np.hypot(dx, dy))
        return -float(min(x, self.width_m - x, y, self.height_m - y))

    def is_inside(
        self,
        px: float,
        py: float,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> bool:
        return self.distance_to_boundary_m(
            px, py, frame_width, frame_height
        ) <= 0.0


def calibrate_from_markers(
    camera_id: str,
    detections: list[MarkerDetection],
    marker_ids: list[int],
    width_m: float,
    height_m: float,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> Calibration:
    """Build a normalized-image-to-metre homography.

    ``marker_ids`` and returned centers are ordered TL, TR, BR, BL.
    """
    parsed_ids = _validate_marker_ids(marker_ids)
    width_m = _positive_finite(width_m, "width_m")
    height_m = _positive_finite(height_m, "height_m")

    detections_by_id: dict[int, list[MarkerDetection]] = {}
    for detection in detections:
        detections_by_id.setdefault(int(detection.marker_id), []).append(detection)
    missing = [marker_id for marker_id in parsed_ids if marker_id not in detections_by_id]
    if missing:
        raise CalibrationGeometryError(f"Markers not detected in frame: {missing}")
    duplicates = [
        marker_id for marker_id in parsed_ids
        if len(detections_by_id[marker_id]) != 1
    ]
    if duplicates:
        raise CalibrationGeometryError(
            f"Markers detected more than once in frame: {duplicates}"
        )

    marker_areas_px: list[float] = []
    for marker_id in parsed_ids:
        corners = np.asarray(
            detections_by_id[marker_id][0].corners,
            dtype=np.float64,
        )
        if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            raise CalibrationGeometryError(
                f"Marker {marker_id} must have exactly 4 finite corners."
            )
        marker_area = abs(_polygon_area(corners))
        if marker_area < MIN_MARKER_AREA_PX:
            raise CalibrationGeometryError(
                f"Marker {marker_id} is too small: area {marker_area:.2f}px², "
                f"required at least {MIN_MARKER_AREA_PX:.2f}px²."
            )
        marker_areas_px.append(marker_area)

    centers_px = np.asarray(
        [detections_by_id[marker_id][0].center for marker_id in parsed_ids],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(centers_px)):
        raise CalibrationGeometryError("Marker centers must contain finite coordinates.")

    if frame_width is None or frame_height is None:
        # Backward-compatible inference for direct callers. API routes always
        # pass the decoded frame dimensions.
        inferred_width = max(1, int(math.ceil(float(np.max(centers_px[:, 0])))))
        inferred_height = max(1, int(math.ceil(float(np.max(centers_px[:, 1])))))
        frame_width = inferred_width if frame_width is None else frame_width
        frame_height = inferred_height if frame_height is None else frame_height
    frame_width, frame_height = _frame_dimensions(frame_width, frame_height)

    normalized = centers_px / np.asarray([frame_width, frame_height], dtype=np.float64)
    outside = [
        parsed_ids[index]
        for index, (u, v) in enumerate(normalized)
        if u < -1e-9 or u > 1.0 + 1e-9 or v < -1e-9 or v > 1.0 + 1e-9
    ]
    if outside:
        raise CalibrationGeometryError(
            f"Marker centers outside source frame {frame_width}x{frame_height}: {outside}"
        )

    # TL→TR→BR→BL in image coordinates (Y grows down) must have positive,
    # consistently convex turns and the expected broad left/right/top/bottom
    # ordering. These checks catch swapped IDs and bow-tie quadrilaterals.
    signed_area = _polygon_area(normalized)
    if signed_area <= 0.0:
        raise CalibrationGeometryError(
            "Marker order/winding is invalid; expected TL, TR, BR, BL."
        )
    turns = []
    for index in range(4):
        edge_a = normalized[(index + 1) % 4] - normalized[index]
        edge_b = normalized[(index + 2) % 4] - normalized[(index + 1) % 4]
        turns.append(float(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]))
    if min(turns) <= 1e-8:
        raise CalibrationGeometryError(
            "Marker quadrilateral must be strictly convex and non-self-intersecting."
        )
    tl, tr, br, bl = normalized
    if tr[0] <= tl[0] or br[0] <= bl[0] or bl[1] <= tl[1] or br[1] <= tr[1]:
        raise CalibrationGeometryError(
            "Marker positions do not match TL, TR, BR, BL semantics."
        )

    edges = [
        float(np.linalg.norm(normalized[(i + 1) % 4] - normalized[i]))
        for i in range(4)
    ]
    coverage = signed_area
    angles = _interior_angles(normalized)
    if coverage < MIN_COVERAGE_RATIO:
        raise CalibrationGeometryError(
            "Reference quadrilateral is too small: "
            f"coverage {coverage:.4f}, required at least {MIN_COVERAGE_RATIO:.4f}."
        )
    if min(edges) < MIN_EDGE_RATIO:
        raise CalibrationGeometryError(
            "Reference edge is too short: "
            f"{min(edges):.4f}, required at least {MIN_EDGE_RATIO:.4f}."
        )
    if min(angles) < MIN_INTERIOR_ANGLE_DEG or max(angles) > MAX_INTERIOR_ANGLE_DEG:
        raise CalibrationGeometryError(
            "Reference quadrilateral is too skewed: interior angles "
            f"{min(angles):.1f}°..{max(angles):.1f}°, accepted range "
            f"{MIN_INTERIOR_ANGLE_DEG:.1f}°..{MAX_INTERIOR_ANGLE_DEG:.1f}°."
        )

    world = np.asarray(
        [
            (0.0, 0.0),
            (width_m, 0.0),
            (width_m, height_m),
            (0.0, height_m),
        ],
        dtype=np.float64,
    )
    homography, _status = cv2.findHomography(normalized, world, method=0)
    if homography is None or not np.all(np.isfinite(homography)):
        raise CalibrationGeometryError(
            "Homography could not be computed from the marker geometry."
        )
    if abs(float(homography[2, 2])) <= 1e-12:
        raise CalibrationGeometryError("Homography has an invalid scale.")
    homography = homography / homography[2, 2]
    determinant = float(np.linalg.det(homography))
    condition = float(np.linalg.cond(homography))
    if abs(determinant) <= 1e-12 or not math.isfinite(condition):
        raise CalibrationGeometryError("Homography is singular or non-finite.")
    if condition > MAX_HOMOGRAPHY_CONDITION:
        raise CalibrationGeometryError(
            "Homography is too ill-conditioned for metric measurement: "
            f"condition number {condition:.3g}."
        )

    quality = _geometry_quality(
        normalized,
        homography,
        world,
        min(marker_areas_px),
    )
    return Calibration(
        camera_id=camera_id,
        marker_ids=parsed_ids,
        width_m=width_m,
        height_m=height_m,
        homography=homography.tolist(),
        source_frame_width=frame_width,
        source_frame_height=frame_height,
        source_aspect_ratio=frame_width / frame_height,
        marker_centers=normalized.tolist(),
        marker_centers_px=centers_px.tolist(),
        quality=quality,
    )


class CalibrationStore:
    def __init__(self, path: str | None = None):
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, Calibration] = {}
        if path and os.path.exists(path):
            self._load()

    def _load(self) -> None:
        try:
            with open(self._path, "r", encoding="utf-8") as file:
                raw = json.load(file)
        except (OSError, ValueError, TypeError):
            self._data = {}
            return
        if not isinstance(raw, dict):
            self._data = {}
            return

        known_fields = {item.name for item in fields(Calibration)}
        loaded: dict[str, Calibration] = {}
        for camera_id, obj in raw.items():
            if not isinstance(camera_id, str) or not isinstance(obj, dict):
                continue
            try:
                obj = dict(obj)
                if (
                    "source_aspect_ratio" not in obj
                    and "source_frame_aspect" in obj
                ):
                    obj["source_aspect_ratio"] = obj["source_frame_aspect"]
                values = {key: value for key, value in obj.items() if key in known_fields}
                values["camera_id"] = camera_id
                # Missing source geometry marks a legacy pixel-space record.
                values.setdefault("source_frame_width", 0)
                values.setdefault("source_frame_height", 0)
                values.setdefault("source_aspect_ratio", 0.0)
                values.setdefault("marker_centers", [])
                values.setdefault("marker_centers_px", [])
                values.setdefault("quality", _legacy_quality())
                loaded[camera_id] = Calibration(**values)
            except (CalibrationError, TypeError, ValueError):
                # One corrupt camera record must not erase other valid entries.
                continue
        self._data = loaded

    def _save(self) -> None:
        if not self._path:
            return
        directory = os.path.dirname(self._path) or "."
        os.makedirs(directory, exist_ok=True)
        temporary_path = f"{self._path}.tmp"
        try:
            with open(temporary_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        camera_id: asdict(calibration)
                        for camera_id, calibration in self._data.items()
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, self._path)
        finally:
            # os.replace removes the source on success. On failure, do not
            # leave a partial temporary record beside the durable store.
            if os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass

    def get(self, camera_id: str) -> Calibration | None:
        with self._lock:
            return self._data.get(camera_id)

    def set(self, calibration: Calibration) -> Calibration:
        with self._lock:
            previous = self._data.get(calibration.camera_id)
            self._data[calibration.camera_id] = calibration
            try:
                self._save()
            except Exception:
                if previous is None:
                    self._data.pop(calibration.camera_id, None)
                else:
                    self._data[calibration.camera_id] = previous
                raise
            return calibration

    def clear(self, camera_id: str) -> bool:
        with self._lock:
            removed = self._data.pop(camera_id, None)
            if removed is None:
                return False
            try:
                self._save()
            except Exception:
                self._data[camera_id] = removed
                raise
            return True

    def all(self) -> list[Calibration]:
        with self._lock:
            return list(self._data.values())
