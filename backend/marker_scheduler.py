"""Rate-limited, per-camera ArUco detection.

Marker detection is useful only for marker-defined zones and while the
calibration page is open.  Keeping that policy in one small component avoids
running OpenCV ArUco on every uploaded frame while still returning the most
recent result between scheduled samples.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import threading
import time
from typing import Callable

import numpy as np

from backend.marker_detector import MarkerDetection, MarkerDetector


def _copy_detection(detection: MarkerDetection) -> MarkerDetection:
    return MarkerDetection(
        marker_id=int(detection.marker_id),
        # The ArUco contract is exactly four corners.  Truncation also keeps
        # the cache strictly bounded if a custom detector violates it.
        corners=[(float(x), float(y)) for x, y in detection.corners[:4]],
    )


def _copy_detections(
    detections: tuple[MarkerDetection, ...] | list[MarkerDetection],
) -> list[MarkerDetection]:
    # MarkerDetection contains a mutable corners list.  Never expose the
    # scheduler's cached objects to callers.
    return [_copy_detection(detection) for detection in detections]


@dataclass
class _CameraState:
    detections: tuple[MarkerDetection, ...] = field(default_factory=tuple)
    last_detection_at: float | None = None
    calibration_active_until: float = 0.0


class MarkerScheduler:
    """Thread-safe, bounded scheduler around :class:`MarkerDetector`.

    Normal processing is capped at ``normal_fps`` and is disabled unless the
    camera has an active marker-defined zone.  A calibration preview call
    touches a short-lived session; while it is alive, ingest may sample at
    ``calibration_fps`` even without marker zones.  ``force=True`` is reserved
    for an explicit calibration submission and performs one immediate scan of
    that exact frame.
    """

    def __init__(
        self,
        detector: MarkerDetector,
        *,
        normal_fps: float = 1.0,
        calibration_fps: float = 5.0,
        calibration_session_ttl: float = 3.0,
        max_cameras: int = 16,
        max_markers_per_camera: int = 64,
        clock: Callable[[], float] = time.monotonic,
    ):
        if normal_fps < 0.0 or calibration_fps <= 0.0:
            raise ValueError("marker scheduler FPS values must be positive")
        if calibration_session_ttl <= 0.0:
            raise ValueError("calibration_session_ttl must be positive")
        if max_cameras < 1 or max_markers_per_camera < 1:
            raise ValueError("marker scheduler bounds must be positive")

        self._detector = detector
        self._normal_interval = (
            1.0 / float(normal_fps) if normal_fps > 0.0 else float("inf")
        )
        self._calibration_interval = 1.0 / float(calibration_fps)
        self._calibration_session_ttl = float(calibration_session_ttl)
        self._max_cameras = int(max_cameras)
        self._max_markers_per_camera = int(max_markers_per_camera)
        self._clock = clock
        self._lock = threading.RLock()
        self._states: OrderedDict[str, _CameraState] = OrderedDict()

    def _state_locked(self, camera_id: str) -> _CameraState:
        camera_id = str(camera_id)
        state = self._states.get(camera_id)
        if state is None:
            state = _CameraState()
            self._states[camera_id] = state
            while len(self._states) > self._max_cameras:
                self._states.popitem(last=False)
        else:
            self._states.move_to_end(camera_id)
        return state

    def touch_calibration_session(
        self,
        camera_id: str,
        *,
        now: float | None = None,
    ) -> None:
        """Keep calibration sampling enabled without adding a new API route.

        The existing metadata-preview endpoint calls this method.  Its regular
        polling acts as a lease which naturally expires after the tab is
        closed or the selected camera changes.
        """
        sample_time = self._clock() if now is None else float(now)
        with self._lock:
            state = self._state_locked(camera_id)
            state.calibration_active_until = max(
                state.calibration_active_until,
                sample_time + self._calibration_session_ttl,
            )

    def calibration_session_active(
        self,
        camera_id: str,
        *,
        now: float | None = None,
    ) -> bool:
        sample_time = self._clock() if now is None else float(now)
        with self._lock:
            state = self._states.get(str(camera_id))
            return bool(
                state is not None
                and state.calibration_active_until > sample_time
            )

    def get_cached(self, camera_id: str) -> list[MarkerDetection]:
        with self._lock:
            state = self._states.get(str(camera_id))
            if state is None:
                return []
            self._states.move_to_end(str(camera_id))
            return _copy_detections(state.detections)

    def process(
        self,
        camera_id: str,
        frame: np.ndarray,
        *,
        marker_zones_active: bool = False,
        calibration: bool = False,
        force: bool = False,
        now: float | None = None,
    ) -> list[MarkerDetection]:
        """Return a fresh or cached detection list for ``camera_id``.

        ``calibration=True`` both renews the calibration lease and selects the
        5 FPS interval.  An already-active lease has the same effect for calls
        made by the normal ingest path.  With neither a marker zone nor such a
        lease, no detector work is performed and an empty public result is
        returned (the bounded internal cache is retained for a later sample).
        """
        sample_time = self._clock() if now is None else float(now)
        with self._lock:
            state = self._state_locked(camera_id)
            if calibration or force:
                state.calibration_active_until = max(
                    state.calibration_active_until,
                    sample_time + self._calibration_session_ttl,
                )

            calibration_active = (
                calibration
                or force
                or state.calibration_active_until > sample_time
            )
            enabled = bool(marker_zones_active or calibration_active)
            if not enabled:
                return []

            interval = (
                self._calibration_interval
                if calibration_active
                else self._normal_interval
            )
            due = (
                state.last_detection_at is None
                or sample_time - state.last_detection_at >= interval
            )
            if not force and not due:
                return _copy_detections(state.detections)

            # MarkerDetector owns a reusable OpenCV detector.  Keeping the
            # call under the scheduler lock serializes concurrent preview and
            # ingest requests and makes the whole wrapper thread-safe.
            detected = self._detector.detect(frame)
            bounded = tuple(
                _copy_detection(item)
                for item in detected[: self._max_markers_per_camera]
            )
            state.detections = bounded
            state.last_detection_at = sample_time
            return _copy_detections(bounded)

    @property
    def camera_count(self) -> int:
        with self._lock:
            return len(self._states)
