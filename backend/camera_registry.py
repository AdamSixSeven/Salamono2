"""Bounded LRU registry for transient live-camera status."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any


class CameraRegistry(OrderedDict[str, dict[str, Any]]):
    """Dictionary-compatible status registry with a hard camera bound.

    Persisted calibrations are stored separately, so evicting an old live
    status row never removes calibration data. Reassigning a camera refreshes
    its LRU position.
    """

    def __init__(self, *args, max_cameras: int = 32, **kwargs):
        if max_cameras < 1:
            raise ValueError("max_cameras must be at least 1")
        self.max_cameras = int(max_cameras)
        super().__init__()
        if args or kwargs:
            self.update(*args, **kwargs)

    def __setitem__(self, camera_id: str, value: dict[str, Any]) -> None:
        camera_key = str(camera_id)
        if camera_key in self:
            super().__delitem__(camera_key)
        super().__setitem__(camera_key, value)
        while len(self) > self.max_cameras:
            self.popitem(last=False)

    def update(self, *args, **kwargs) -> None:
        incoming = dict(*args, **kwargs)
        for camera_id, value in incoming.items():
            self[camera_id] = value
