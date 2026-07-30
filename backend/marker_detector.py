"""ArUco marker detection.

Michał Salamonowicz (2026-07-08): markery na taśmach spełniają trzy role
naraz — oznaczają strefę niebezpieczną, kalibrują kamerę i pozwalają
mierzyć odległość pracownika od strefy w metrach.

Używamy dict 4x4_50 (50 markerów po 4x4 bity) — tania taśma A4 czytelna
z odległości ~10m.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# 4x4_50: 50 unique markers, 4x4-bit payload. Bigger dicts (5x5, 6x6)
# work at longer distance but need larger printed markers.
DEFAULT_DICT = cv2.aruco.DICT_4X4_50


@dataclass
class MarkerDetection:
    marker_id: int
    corners: list[tuple[float, float]]  # 4 corners, TL → TR → BR → BL, in px

    @property
    def center(self) -> tuple[float, float]:
        xs = [c[0] for c in self.corners]
        ys = [c[1] for c in self.corners]
        return (sum(xs) / 4.0, sum(ys) / 4.0)


class MarkerDetector:
    def __init__(self, dictionary_id: int = DEFAULT_DICT):
        self._dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self._params = cv2.aruco.DetectorParameters()
        # OpenCV 4.7+ exposes ArucoDetector; older builds expose the free
        # function cv2.aruco.detectMarkers. Prefer the class when available.
        self._detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector = cv2.aruco.ArucoDetector(self._dictionary, self._params)

    def detect(self, frame_bgr: np.ndarray) -> list[MarkerDetection]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._dictionary, parameters=self._params,
            )
        if ids is None or len(ids) == 0:
            return []
        out: list[MarkerDetection] = []
        for i, mid in enumerate(ids.flatten().tolist()):
            pts = corners[i].reshape(-1, 2)
            out.append(MarkerDetection(
                marker_id=int(mid),
                corners=[(float(x), float(y)) for x, y in pts],
            ))
        return out
