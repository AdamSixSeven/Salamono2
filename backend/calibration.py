"""Camera calibration from 4 ArUco markers.

Idea (spotkanie 2026-07-08, Michał S. + Adam W.): rozstawiasz 4 markery na
taśmach w rogach obszaru referencyjnego (np. 3 m × 3 m). Backend liczy
homografię pixel → metry i potem dla każdego wykrytego pracownika zwraca
odległość jego stóp od granic tego obszaru w metrach realnych.

To jest "killer feature" pod pitch — z tego wynika też realny warning
distance dla stref niebezpiecznych.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from backend.marker_detector import MarkerDetection


@dataclass
class Calibration:
    """Homography from image pixels to a metric plane (X, Y in metres).

    marker_ids: [TL, TR, BR, BL] — order matches width_m/height_m axes.
    """
    camera_id: str
    marker_ids: list[int]
    width_m: float
    height_m: float
    # 3x3 homography, row-major (JSON-safe list of lists).
    homography: list[list[float]]
    created_at: float = field(default_factory=time.time)

    def project(self, px: float, py: float) -> tuple[float, float]:
        H = np.asarray(self.homography, dtype=np.float64)
        vec = H @ np.array([px, py, 1.0])
        if abs(vec[2]) < 1e-9:
            return (0.0, 0.0)
        return (float(vec[0] / vec[2]), float(vec[1] / vec[2]))

    def distance_to_boundary_m(self, px: float, py: float) -> float:
        """Signed distance to reference-rectangle boundary in metres.

        Positive = outside the rectangle (distance to nearest edge).
        Negative = inside the rectangle (negative distance to nearest edge).
        Zero = exactly on the edge.
        """
        x, y = self.project(px, py)
        w, h = self.width_m, self.height_m
        dx = max(0.0 - x, x - w, 0.0)  # 0 if inside x-range
        dy = max(0.0 - y, y - h, 0.0)  # 0 if inside y-range
        if dx > 0 or dy > 0:
            return float(np.hypot(dx, dy))
        # Inside — negative distance to nearest edge.
        inside_gap = min(x, w - x, y, h - y)
        return -float(inside_gap)

    def is_inside(self, px: float, py: float) -> bool:
        return self.distance_to_boundary_m(px, py) <= 0.0


def calibrate_from_markers(
    camera_id: str,
    detections: list[MarkerDetection],
    marker_ids: list[int],
    width_m: float,
    height_m: float,
) -> Calibration:
    """Compute homography from 4 markers arranged TL, TR, BR, BL.

    Uses each marker's centre in image space to solve the homography that
    maps image pixels to the reference plane (0,0)→(width_m, height_m).
    """
    if len(marker_ids) != 4:
        raise ValueError("Need exactly 4 marker IDs (TL, TR, BR, BL).")
    if width_m <= 0 or height_m <= 0:
        raise ValueError("width_m and height_m must be positive.")
    by_id = {d.marker_id: d for d in detections}
    missing = [mid for mid in marker_ids if mid not in by_id]
    if missing:
        raise ValueError(f"Markers not detected in frame: {missing}")
    image_pts = np.array([by_id[mid].center for mid in marker_ids],
                         dtype=np.float64)
    world_pts = np.array([
        (0.0, 0.0),
        (width_m, 0.0),
        (width_m, height_m),
        (0.0, height_m),
    ], dtype=np.float64)
    H, status = cv2.findHomography(image_pts, world_pts)
    if H is None:
        raise ValueError("Homography could not be computed (degenerate markers?).")
    return Calibration(
        camera_id=camera_id,
        marker_ids=list(marker_ids),
        width_m=float(width_m),
        height_m=float(height_m),
        homography=H.tolist(),
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
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for cam, obj in raw.items():
                self._data[cam] = Calibration(**obj)
        except Exception:
            self._data = {}

    def _save(self) -> None:
        if not self._path:
            return
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({cam: asdict(cal) for cam, cal in self._data.items()},
                      f, ensure_ascii=False, indent=2)

    def get(self, camera_id: str) -> Calibration | None:
        with self._lock:
            return self._data.get(camera_id)

    def set(self, cal: Calibration) -> Calibration:
        with self._lock:
            self._data[cal.camera_id] = cal
            self._save()
            return cal

    def clear(self, camera_id: str) -> bool:
        with self._lock:
            removed = self._data.pop(camera_id, None) is not None
            if removed:
                self._save()
            return removed

    def all(self) -> list[Calibration]:
        with self._lock:
            return list(self._data.values())
