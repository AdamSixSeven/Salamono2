"""Thread-safe, bounded snapshots of the latest unannotated camera frames."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LatestFrame:
    """An owned BGR snapshot captured before any detection overlays."""

    camera_id: str
    timestamp: float
    received_at: float
    frame: np.ndarray
    width: int
    height: int
    mode: str
    image_bytes: bytes | None = None
    content_type: str | None = None

    @property
    def captured_at(self) -> float:
        """Backward-compatible name used by the raw calibration preview."""
        return self.timestamp


class LatestFrameStore:
    """Keep one isolated frame per camera and evict oldest camera entries."""

    def __init__(
        self,
        max_cameras: int = 16,
        max_frame_bytes: int = 12 * 1024 * 1024,
    ):
        if max_cameras < 1:
            raise ValueError("max_cameras must be at least 1")
        if max_frame_bytes < 1:
            raise ValueError("max_frame_bytes must be at least 1")
        self.max_cameras = int(max_cameras)
        self.max_frame_bytes = int(max_frame_bytes)
        self._lock = threading.Lock()
        self._frames: OrderedDict[str, LatestFrame] = OrderedDict()

    def set(
        self,
        camera_id: str,
        frame: np.ndarray,
        timestamp: float,
        mode: str,
        *,
        received_at: float | None = None,
        image_bytes: bytes | bytearray | memoryview | None = None,
        content_type: str | None = None,
    ) -> bool:
        """Store an owned copy of a decoded BGR frame.

        ``frame.copy()`` is deliberately performed before taking the lock:
        downstream detectors may reuse or annotate their input while the
        stored snapshot must remain exactly as it was at ingest time.
        """
        if not camera_id:
            raise ValueError("camera_id must not be empty")
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy array")
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("frame must be a non-empty BGR image")
        if frame.nbytes > self.max_frame_bytes:
            return False
        if not mode:
            raise ValueError("mode must not be empty")

        raw = bytes(image_bytes) if image_bytes is not None else None
        if raw is not None and len(raw) > self.max_frame_bytes:
            # The decoded BGR snapshot is the primary contract. Oversized
            # optional preview bytes must not prevent that frame being stored.
            raw = None

        owned_frame = frame.copy()
        height, width = owned_frame.shape[:2]
        item = LatestFrame(
            camera_id=camera_id,
            timestamp=float(timestamp),
            received_at=float(received_at if received_at is not None else time.time()),
            frame=owned_frame,
            width=int(width),
            height=int(height),
            mode=str(mode),
            image_bytes=raw,
            content_type=(
                str(content_type or "application/octet-stream")
                if raw is not None
                else None
            ),
        )
        with self._lock:
            # Updating a camera also refreshes its eviction order.
            self._frames.pop(camera_id, None)
            self._frames[camera_id] = item
            while len(self._frames) > self.max_cameras:
                self._frames.popitem(last=False)
        return True

    def put(
        self,
        camera_id: str,
        image_bytes: bytes,
        *,
        content_type: str,
        width: int,
        height: int,
        captured_at: float,
        received_at: float | None = None,
        mode: str = "site",
    ) -> bool:
        """Compatibility wrapper for the former raw-bytes-only API.

        Valid encoded images are decoded to BGR.  A blank BGR frame is used
        only for legacy callers that supplied arbitrary bytes; their raw
        preview contract remains unchanged.
        """
        if not image_bytes:
            raise ValueError("image_bytes must not be empty")
        if len(image_bytes) > self.max_frame_bytes:
            return False
        if width < 1 or height < 1:
            raise ValueError("frame dimensions must be positive")

        raw = bytes(image_bytes)
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            decoded = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        elif decoded.shape[:2] != (int(height), int(width)):
            decoded = cv2.resize(decoded, (int(width), int(height)))

        return self.set(
            camera_id,
            decoded,
            captured_at,
            mode,
            received_at=received_at,
            image_bytes=raw,
            content_type=content_type,
        )

    def get(
        self,
        camera_id: str,
        *,
        max_age_seconds: float | None = None,
        now: float | None = None,
    ) -> LatestFrame | None:
        """Return an isolated snapshot, optionally requiring a recent receive."""
        with self._lock:
            item = self._frames.get(camera_id)
            if item is None:
                return None
            if max_age_seconds is not None:
                current = float(now if now is not None else time.time())
                if current - item.received_at > max_age_seconds:
                    return None

        # Stored arrays are never mutated after insertion, so the relatively
        # expensive image copy can safely happen without holding the lock.
        return LatestFrame(
            camera_id=item.camera_id,
            timestamp=item.timestamp,
            received_at=item.received_at,
            frame=item.frame.copy(),
            width=item.width,
            height=item.height,
            mode=item.mode,
            image_bytes=item.image_bytes,
            content_type=item.content_type,
        )

    def remove(self, camera_id: str) -> None:
        with self._lock:
            self._frames.pop(str(camera_id), None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)
