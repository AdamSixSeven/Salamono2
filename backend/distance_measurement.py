"""Ground-plane person/machine distance measurement."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from backend.detector import Detection


@dataclass(frozen=True)
class DistancePair:
    person_index: int
    hazard_index: int
    person_class: str
    hazard_class: str
    person_confidence: float
    hazard_confidence: float
    person_box: tuple[int, int, int, int]
    hazard_box: tuple[int, int, int, int]
    distance_m: float
    person_point_px: tuple[int, int]
    hazard_point_px: tuple[int, int]


class GroundPlaneDistanceService:
    """Project bbox ground contacts to a calibrated ground plane."""

    def __init__(self) -> None:
        self.intrinsics_path = Path(
            os.getenv(
                "DISTANCE_INTRINSICS_PATH",
                "distance_assets/camera_intrinsics.npz",
            )
        )
        self.extrinsics_path = Path(
            os.getenv(
                "DISTANCE_EXTRINSICS_PATH",
                "distance_assets/camera_extrinsics.npz",
            )
        )
        self.calibration_width = max(
            1, int(os.getenv("DISTANCE_CALIBRATION_WIDTH", "1920"))
        )
        self.calibration_height = max(
            1, int(os.getenv("DISTANCE_CALIBRATION_HEIGHT", "1080"))
        )
        # Calibration coordinates use 100 units per metre by default.
        self.units_per_meter = float(os.getenv("DISTANCE_UNITS_PER_METER", "100"))
        self.aspect_tolerance = max(
            0.0, float(os.getenv("DISTANCE_ASPECT_TOLERANCE", "0.02"))
        )

        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None
        self.rvec: np.ndarray | None = None
        self.tvec: np.ndarray | None = None
        self.rotation: np.ndarray | None = None
        self.error: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            if self.units_per_meter <= 0 or not math.isfinite(self.units_per_meter):
                raise ValueError("DISTANCE_UNITS_PER_METER must be positive")
            intr = np.load(self.intrinsics_path)
            ext = np.load(self.extrinsics_path)
            camera_matrix = np.asarray(intr["camera_matrix"], dtype=np.float64)
            dist_coeffs = np.asarray(intr["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
            rvec = np.asarray(ext["rvec"], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(ext["tvec"], dtype=np.float64).reshape(3, 1)
            if camera_matrix.shape != (3, 3):
                raise ValueError("camera_matrix must be 3x3")
            if not all(np.all(np.isfinite(x)) for x in (camera_matrix, dist_coeffs, rvec, tvec)):
                raise ValueError("calibration contains non-finite values")
            rotation, _ = cv2.Rodrigues(rvec)
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.rvec = rvec
            self.tvec = tvec
            self.rotation = rotation
            self.error = None
        except Exception as exc:  # optional module must fail closed
            self.camera_matrix = None
            self.dist_coeffs = None
            self.rvec = None
            self.tvec = None
            self.rotation = None
            self.error = str(exc)

    @property
    def available(self) -> bool:
        return all(
            item is not None
            for item in (
                self.camera_matrix,
                self.dist_coeffs,
                self.rvec,
                self.tvec,
                self.rotation,
            )
        )

    def _frame_compatibility(self, width: int, height: int) -> tuple[bool, str | None]:
        if width <= 0 or height <= 0:
            return False, "Nieprawidłowe wymiary klatki."
        calibrated_aspect = self.calibration_width / self.calibration_height
        frame_aspect = width / height
        delta = abs(frame_aspect - calibrated_aspect) / calibrated_aspect
        if delta > self.aspect_tolerance:
            return (
                False,
                "Niezgodny aspect ratio: kalibracja "
                f"{self.calibration_width}x{self.calibration_height}, klatka {width}x{height} "
                f"({delta * 100:.2f}% > {self.aspect_tolerance * 100:.2f}%).",
            )
        return True, None

    def status(self, width: int | None = None, height: int | None = None) -> dict:
        compatible = None
        warning = None
        if width is not None and height is not None and self.available:
            compatible, warning = self._frame_compatibility(int(width), int(height))
        return {
            "available": self.available,
            "error": self.error,
            "intrinsics_path": str(self.intrinsics_path),
            "extrinsics_path": str(self.extrinsics_path),
            "calibration_width": self.calibration_width,
            "calibration_height": self.calibration_height,
            "units_per_meter": self.units_per_meter,
            "compatible": compatible,
            "warning": warning,
        }

    def _scaled_camera_matrix(self, width: int, height: int) -> np.ndarray:
        if self.camera_matrix is None:
            raise RuntimeError("distance calibration unavailable")
        sx = float(width) / float(self.calibration_width)
        sy = float(height) / float(self.calibration_height)
        matrix = self.camera_matrix.copy()
        matrix[0, 0] *= sx
        matrix[0, 2] *= sx
        matrix[1, 1] *= sy
        matrix[1, 2] *= sy
        return matrix

    def _undistorted_pixel(self, u: float, v: float, matrix: np.ndarray) -> tuple[float, float]:
        assert self.dist_coeffs is not None
        point = np.array([[[float(u), float(v)]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(
            point,
            matrix,
            self.dist_coeffs,
            P=matrix,
        )[0, 0]
        return float(undistorted[0]), float(undistorted[1])

    def ground_point(self, u: float, v: float, width: int, height: int) -> np.ndarray | None:
        if not self.available:
            return None
        compatible, _ = self._frame_compatibility(width, height)
        if not compatible:
            return None
        assert self.rotation is not None and self.tvec is not None
        matrix = self._scaled_camera_matrix(width, height)
        u2, v2 = self._undistorted_pixel(u, v, matrix)
        ray = np.linalg.inv(matrix) @ np.array([[u2], [v2], [1.0]], dtype=np.float64)
        system = np.column_stack(
            (-self.rotation[:, 0], -self.rotation[:, 1], ray[:, 0])
        )
        try:
            solved = np.linalg.solve(system, self.tvec[:, 0])
        except np.linalg.LinAlgError:
            return None
        x, y, ray_scale = (float(solved[0]), float(solved[1]), float(solved[2]))
        if not all(math.isfinite(value) for value in (x, y, ray_scale)):
            return None
        if ray_scale <= 0:
            return None
        return np.array([x, y], dtype=np.float64)

    def image_point(self, xy: np.ndarray, width: int, height: int) -> tuple[int, int] | None:
        if not self.available:
            return None
        assert self.rvec is not None and self.tvec is not None and self.dist_coeffs is not None
        world = np.array([[[float(xy[0]), float(xy[1]), 0.0]]], dtype=np.float64)
        matrix = self._scaled_camera_matrix(width, height)
        projected, _ = cv2.projectPoints(
            world,
            self.rvec,
            self.tvec,
            matrix,
            self.dist_coeffs,
        )
        u, v = projected[0, 0]
        if not math.isfinite(float(u)) or not math.isfinite(float(v)):
            return None
        return int(round(float(u))), int(round(float(v)))

    @staticmethod
    def _point_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray):
        segment = end - start
        denominator = float(np.dot(segment, segment))
        if denominator < 1e-9:
            return float(np.linalg.norm(point - start)), start
        t = float(np.dot(point - start, segment) / denominator)
        t = max(0.0, min(1.0, t))
        projection = start + t * segment
        return float(np.linalg.norm(point - projection)), projection

    @classmethod
    def _segments_distance(
        cls,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        p4: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        # Robust 2-D segment intersection first.
        r = p2 - p1
        s = p4 - p3
        cross = float(r[0] * s[1] - r[1] * s[0])
        qmp = p3 - p1
        if abs(cross) > 1e-9:
            t = float((qmp[0] * s[1] - qmp[1] * s[0]) / cross)
            u = float((qmp[0] * r[1] - qmp[1] * r[0]) / cross)
            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                intersection = p1 + t * r
                return 0.0, intersection, intersection

        d1, q1 = cls._point_to_segment(p1, p3, p4)
        d2, q2 = cls._point_to_segment(p2, p3, p4)
        d3, q3 = cls._point_to_segment(p3, p1, p2)
        d4, q4 = cls._point_to_segment(p4, p1, p2)
        return min(
            (d1, p1, q1),
            (d2, p2, q2),
            (d3, q3, p3),
            (d4, q4, p4),
            key=lambda item: item[0],
        )

    def measure(
        self,
        detections: Iterable[Detection],
        width: int,
        height: int,
        *,
        max_distance_m: float | None = None,
    ) -> tuple[list[DistancePair], int, int]:
        if not self.available:
            return [], 0, 0
        compatible, _ = self._frame_compatibility(width, height)
        if not compatible:
            return [], 0, 0

        persons = [item for item in detections if item.category == "person"]
        hazards = [
            item
            for item in detections
            if item.category in {"vehicle", "hazard", "danger"}
        ]
        pairs: list[DistancePair] = []

        for person_index, person in enumerate(persons):
            px1, _py1, px2, py2 = [int(value) for value in person.box]
            person_a = self.ground_point(px1, py2, width, height)
            person_b = self.ground_point(px2, py2, width, height)
            if person_a is None or person_b is None:
                continue

            for hazard_index, hazard in enumerate(hazards):
                hx1, _hy1, hx2, hy2 = [int(value) for value in hazard.box]
                hazard_a = self.ground_point(hx1, hy2, width, height)
                hazard_b = self.ground_point(hx2, hy2, width, height)
                if hazard_a is None or hazard_b is None:
                    continue

                distance_units, person_xy, hazard_xy = self._segments_distance(
                    person_a, person_b, hazard_a, hazard_b
                )
                distance_m = distance_units / self.units_per_meter
                if max_distance_m is not None and distance_m > max_distance_m:
                    continue
                person_px = self.image_point(person_xy, width, height)
                hazard_px = self.image_point(hazard_xy, width, height)
                if person_px is None or hazard_px is None:
                    continue
                pairs.append(
                    DistancePair(
                        person_index=person_index,
                        hazard_index=hazard_index,
                        person_class=str(person.class_name),
                        hazard_class=str(hazard.class_name),
                        person_confidence=float(person.confidence),
                        hazard_confidence=float(hazard.confidence),
                        person_box=tuple(int(value) for value in person.box),
                        hazard_box=tuple(int(value) for value in hazard.box),
                        distance_m=float(distance_m),
                        person_point_px=person_px,
                        hazard_point_px=hazard_px,
                    )
                )

        pairs.sort(key=lambda item: item.distance_m)
        return pairs, len(persons), len(hazards)


__all__ = ["DistancePair", "GroundPlaneDistanceService"]
