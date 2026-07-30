"""Thread-safe per-camera runtime switches for expensive analysis modules."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import threading
import time


@dataclass(frozen=True)
class RuntimeProcessingOptions:
    boxes: bool = True
    posture: bool = True
    zones: bool = True
    markers: bool = True
    distances: bool = True
    worker_id: bool = True
    ppe: bool = True
    updated_at: float = 0.0

    def requires_detector(self, mode: str) -> bool:
        if str(mode).lower() == "checkpoint":
            return bool(self.ppe or self.worker_id)
        return bool(
            self.boxes
            or self.posture
            or self.zones
            or self.distances
            or self.worker_id
        )

    def enabled_modules(self) -> list[str]:
        return [
            name
            for name in (
                "boxes",
                "posture",
                "zones",
                "markers",
                "distances",
                "worker_id",
                "ppe",
            )
            if bool(getattr(self, name))
        ]

    def to_dict(self) -> dict[str, bool | float]:
        return asdict(self)


class RuntimeProcessingStore:
    """In-memory switches updated by the live panel without restarting FastAPI."""

    _FIELDS = {
        "boxes",
        "posture",
        "zones",
        "markers",
        "distances",
        "worker_id",
        "ppe",
    }

    def __init__(self, *, max_cameras: int = 64):
        self.max_cameras = max(1, int(max_cameras))
        self._lock = threading.RLock()
        self._items: dict[str, RuntimeProcessingOptions] = {}
        self._last_used: dict[str, float] = {}

    def get(self, camera_id: str) -> RuntimeProcessingOptions:
        key = str(camera_id or "cam_default")
        with self._lock:
            item = self._items.get(key)
            if item is None:
                item = RuntimeProcessingOptions(updated_at=time.time())
                self._items[key] = item
            self._last_used[key] = time.monotonic()
            self._evict_locked()
            return replace(item)

    def update(self, camera_id: str, **changes: bool) -> RuntimeProcessingOptions:
        key = str(camera_id or "cam_default")
        unknown = set(changes) - self._FIELDS
        if unknown:
            raise ValueError(f"Unsupported runtime options: {', '.join(sorted(unknown))}")
        with self._lock:
            current = self._items.get(key) or RuntimeProcessingOptions()
            normalized = {
                field: bool(value)
                for field, value in changes.items()
                if field in self._FIELDS
            }
            updated = replace(current, **normalized, updated_at=time.time())
            self._items[key] = updated
            self._last_used[key] = time.monotonic()
            self._evict_locked()
            return replace(updated)

    def reset(self, camera_id: str) -> RuntimeProcessingOptions:
        key = str(camera_id or "cam_default")
        with self._lock:
            self._items.pop(key, None)
            self._last_used.pop(key, None)
        return self.get(key)

    def _evict_locked(self) -> None:
        while len(self._items) > self.max_cameras:
            oldest = min(self._last_used, key=self._last_used.get)
            self._items.pop(oldest, None)
            self._last_used.pop(oldest, None)
