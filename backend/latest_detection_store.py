"""Bounded exact-frame YOLO cache shared with optional depth inference."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import threading

from backend.detector import Detection


@dataclass(frozen=True)
class DetectionSnapshot:
    camera_id: str
    timestamp: float
    width: int
    height: int
    detections: tuple[Detection, ...]


class LatestDetectionStore:
    def __init__(self, *, max_cameras: int = 16):
        self.max_cameras = max(1, int(max_cameras))
        self._items: OrderedDict[str, DetectionSnapshot] = OrderedDict()
        self._lock = threading.RLock()

    def put(
        self,
        camera_id: str,
        timestamp: float,
        width: int,
        height: int,
        detections: list[Detection],
    ) -> None:
        key = str(camera_id)
        snapshot = DetectionSnapshot(
            camera_id=key,
            timestamp=float(timestamp),
            width=int(width),
            height=int(height),
            detections=tuple(replace(item) for item in detections),
        )
        with self._lock:
            self._items.pop(key, None)
            self._items[key] = snapshot
            while len(self._items) > self.max_cameras:
                self._items.popitem(last=False)

    def get_exact(
        self,
        camera_id: str,
        timestamp: float,
        width: int,
        height: int,
        *,
        timestamp_tolerance: float = 1e-6,
    ) -> DetectionSnapshot | None:
        key = str(camera_id)
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            if (
                abs(item.timestamp - float(timestamp)) > timestamp_tolerance
                or item.width != int(width)
                or item.height != int(height)
            ):
                return None
            self._items.move_to_end(key)
            return DetectionSnapshot(
                camera_id=item.camera_id,
                timestamp=item.timestamp,
                width=item.width,
                height=item.height,
                detections=tuple(replace(detection) for detection in item.detections),
            )

    def remove(self, camera_id: str) -> None:
        with self._lock:
            self._items.pop(str(camera_id), None)


__all__ = ["DetectionSnapshot", "LatestDetectionStore"]
