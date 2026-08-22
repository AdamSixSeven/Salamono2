"""Per-camera adaptive YOLO input size for fall-recovery frames."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
import time
from typing import Iterable

from config import YOLOConfig


@dataclass
class _CameraState:
    frame_count: int = 0
    no_person_frames: int = 0
    recovery_frames_remaining: int = 0
    last_img_size: int = 0
    last_reason: str = "base"
    last_trigger: str | None = None
    last_used: float = 0.0
    base_frames: int = 0
    recovery_frames: int = 0


class AdaptiveYoloSizeController:
    """Choose 512 normally and 640 only when fall recovery can add value.

    A second inference is never run for the same frame.  Horizontal person
    boxes and fall/lying posture signals arm a short high-resolution window.
    When YOLO sees no person at all, one periodic recovery scan prevents a
    missed lying person from leaving the pipeline permanently at the base
    resolution.
    """

    def __init__(self, config: YOLOConfig, *, max_cameras: int = 64):
        self.cfg = config
        self.base_size = max(32, int(config.img_size))
        self.recovery_size = max(
            self.base_size,
            int(config.fall_recovery_img_size),
        )
        self.enabled = bool(
            config.adaptive_size_enabled
            and self.recovery_size > self.base_size
        )
        self.scan_interval = max(
            1,
            int(config.fall_recovery_scan_interval_frames),
        )
        self.hold_frames = max(1, int(config.fall_recovery_hold_frames))
        self.horizontal_ratio = max(
            1.0,
            float(config.fall_recovery_horizontal_ratio),
        )
        self.max_cameras = max(1, int(max_cameras))
        self._states: OrderedDict[str, _CameraState] = OrderedDict()
        self._lock = threading.RLock()

    def _state(self, camera_id: str) -> _CameraState:
        key = str(camera_id or "cam_default")
        state = self._states.get(key)
        if state is None:
            state = _CameraState(last_img_size=self.base_size)
            self._states[key] = state
        else:
            self._states.move_to_end(key)
        state.last_used = time.monotonic()
        while len(self._states) > self.max_cameras:
            self._states.popitem(last=False)
        return state

    def select_size(
        self,
        camera_id: str,
        *,
        backend_supports_adaptive_size: bool = True,
    ) -> int:
        with self._lock:
            state = self._state(camera_id)
            state.frame_count += 1
            if not self.enabled:
                state.last_img_size = self.base_size
                state.last_reason = "disabled"
                state.base_frames += 1
                return self.base_size
            if not backend_supports_adaptive_size:
                state.last_img_size = self.base_size
                state.last_reason = "static_backend"
                state.base_frames += 1
                return self.base_size

            if state.recovery_frames_remaining > 0:
                state.recovery_frames_remaining -= 1
                state.last_img_size = self.recovery_size
                state.last_reason = "fall_recovery"
                state.recovery_frames += 1
                return self.recovery_size

            periodic_recovery = bool(
                state.no_person_frames > 0
                and state.no_person_frames % self.scan_interval == 0
            )
            if periodic_recovery:
                state.last_img_size = self.recovery_size
                state.last_reason = "no_person_recovery_scan"
                state.recovery_frames += 1
                return self.recovery_size

            state.last_img_size = self.base_size
            state.last_reason = "base"
            state.base_frames += 1
            return self.base_size

    def observe(
        self,
        camera_id: str,
        detections: Iterable,
        posture_assessments: Iterable = (),
    ) -> None:
        with self._lock:
            state = self._state(camera_id)
            persons = [
                detection
                for detection in detections
                if getattr(detection, "category", None) == "person"
            ]
            recovered_at_high_resolution = bool(
                persons
                and state.last_reason == "no_person_recovery_scan"
            )
            if persons:
                state.no_person_frames = 0
            else:
                state.no_person_frames += 1

            trigger: str | None = (
                "person_recovered_at_high_resolution"
                if recovered_at_high_resolution else None
            )
            for person in persons:
                x1, y1, x2, y2 = getattr(person, "box", (0, 0, 0, 0))
                width = max(0.0, float(x2) - float(x1))
                height = max(1.0, float(y2) - float(y1))
                if width / height >= self.horizontal_ratio:
                    trigger = "horizontal_person_box"
                    break

            if trigger is None:
                for assessment in posture_assessments:
                    text = " ".join([
                        str(getattr(assessment, "behavior_label", "") or ""),
                        str(getattr(assessment, "status", "") or ""),
                        " ".join(
                            str(signal)
                            for signal in (
                                getattr(assessment, "signals", None) or []
                            )
                        ),
                    ]).lower()
                    if any(
                        token in text
                        for token in ("fall", "lying", "upad", "leżą", "lezac")
                    ):
                        trigger = "posture_fall_signal"
                        break

            if trigger is not None and self.enabled:
                state.recovery_frames_remaining = max(
                    state.recovery_frames_remaining,
                    self.hold_frames,
                )
                state.last_trigger = trigger

    def reset_camera(self, camera_id: str) -> None:
        with self._lock:
            self._states.pop(str(camera_id or "cam_default"), None)

    def status(self) -> dict:
        with self._lock:
            cameras = {
                camera_id: {
                    "last_img_size": state.last_img_size,
                    "last_reason": state.last_reason,
                    "last_trigger": state.last_trigger,
                    "no_person_frames": state.no_person_frames,
                    "recovery_frames_remaining": state.recovery_frames_remaining,
                    "base_frames": state.base_frames,
                    "recovery_frames": state.recovery_frames,
                }
                for camera_id, state in self._states.items()
            }
        return {
            "enabled": self.enabled,
            "base_img_size": self.base_size,
            "fall_recovery_img_size": self.recovery_size,
            "no_person_scan_interval_frames": self.scan_interval,
            "recovery_hold_frames": self.hold_frames,
            "horizontal_ratio": self.horizontal_ratio,
            "cameras": cameras,
        }


__all__ = ["AdaptiveYoloSizeController"]
