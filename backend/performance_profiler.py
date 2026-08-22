"""Low-overhead rolling performance telemetry for the image pipeline."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import logging
import math
import statistics
import threading
import time
from typing import Mapping


logger = logging.getLogger(__name__)


PERFORMANCE_STAGES: tuple[str, ...] = (
    "video_decode_ms",
    "frame_prepare_ms",
    "yolo_preprocess_ms",
    "yolo_inference_ms",
    "yolo_postprocess_ms",
    "tracking_ms",
    "posture_crop_ms",
    "mediapipe_ms",
    "optical_flow_ms",
    "tcn_ms",
    "worker_id_submit_ms",
    "marker_detection_ms",
    "depth_ms",
    "overlay_ms",
    "jpeg_encode_ms",
    "websocket_send_ms",
    "total_frame_ms",
    "queue_age_ms",
    "dropped_frames",
)

COHORT_WITH_PERSON = "with_person"
COHORT_WITHOUT_PERSON = "without_person"
COHORT_UNKNOWN = "unknown"
PERFORMANCE_COHORTS = (
    COHORT_WITH_PERSON,
    COHORT_WITHOUT_PERSON,
    COHORT_UNKNOWN,
)


def _cohort(has_person: bool | None) -> str:
    if has_person is True:
        return COHORT_WITH_PERSON
    if has_person is False:
        return COHORT_WITHOUT_PERSON
    return COHORT_UNKNOWN


def _summary(values: list[float]) -> dict[str, int | float | None]:
    if not values:
        return {
            "samples": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    ordered = sorted(values)
    rank = 0.95 * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        p95 = ordered[lower]
    else:
        weight = rank - lower
        p95 = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return {
        "samples": len(values),
        "mean": round(float(statistics.fmean(values)), 3),
        "median": round(float(statistics.median(values)), 3),
        "p95": round(float(p95), 3),
        "max": round(float(max(values)), 3),
    }


@dataclass(slots=True)
class FramePerformanceTrace:
    """Mutable timing envelope carried with one immutable :class:`FrameJob`."""

    profiler: "PerformanceProfiler"
    source: str = "live"
    created_at: float = field(default_factory=time.perf_counter)
    queued_at: float = field(default_factory=time.perf_counter)
    values: dict[str, float] = field(default_factory=dict)
    dropped_frames: int = 0
    has_person: bool | None = None
    finalized: bool = False

    def set(self, stage: str, value: float) -> None:
        if stage not in PERFORMANCE_STAGES or self.finalized:
            return
        numeric = float(value)
        if math.isfinite(numeric) and numeric >= 0.0:
            self.values[stage] = numeric

    def add(self, stage: str, value: float) -> None:
        self.set(stage, self.values.get(stage, 0.0) + float(value))

    def mark_queued(self) -> None:
        self.queued_at = time.perf_counter()

    def finish(self, *, has_person: bool | None = None) -> None:
        if self.finalized:
            return
        self.finalized = True
        self.values.setdefault("dropped_frames", float(self.dropped_frames))
        self.values["total_frame_ms"] = max(
            0.0,
            (time.perf_counter() - self.created_at) * 1000.0,
        )
        cohort_value = self.has_person if has_person is None else has_person
        self.profiler.record_frame(
            self.values,
            has_person=cohort_value,
            source=self.source,
        )


class PerformanceProfiler:
    """Thread-safe bounded statistics with periodic DEBUG summaries."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        window_size: int = 600,
        debug_interval_seconds: float = 10.0,
    ):
        self.enabled = bool(enabled)
        self.window_size = max(10, int(window_size))
        self.debug_interval_seconds = max(1.0, float(debug_interval_seconds))
        self._lock = threading.RLock()
        self._metrics = {
            cohort: {
                stage: deque(maxlen=self.window_size)
                for stage in PERFORMANCE_STAGES
            }
            for cohort in PERFORMANCE_COHORTS
        }
        self._frames = Counter()
        self._sources = Counter()
        self._drops = Counter()
        self._started_at = time.time()
        self._last_debug_at = time.monotonic()

    def new_trace(
        self,
        *,
        source: str = "live",
        video_decode_ms: float | None = None,
    ) -> FramePerformanceTrace:
        trace = FramePerformanceTrace(self, source=str(source or "live"))
        if video_decode_ms is not None:
            trace.set("video_decode_ms", video_decode_ms)
            trace.created_at -= max(0.0, float(video_decode_ms)) / 1000.0
        return trace

    def observe(
        self,
        stage: str,
        value: float,
        *,
        has_person: bool | None,
        source: str = "worker",
    ) -> None:
        if not self.enabled or stage not in PERFORMANCE_STAGES:
            return
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            return
        cohort = _cohort(has_person)
        with self._lock:
            self._metrics[cohort][stage].append(numeric)
            self._sources[str(source or "worker")] += 1

    def record_frame(
        self,
        values: Mapping[str, float],
        *,
        has_person: bool | None,
        source: str,
    ) -> None:
        if not self.enabled:
            return
        cohort = _cohort(has_person)
        should_log = False
        with self._lock:
            for stage, value in values.items():
                if stage not in PERFORMANCE_STAGES:
                    continue
                numeric = float(value)
                if math.isfinite(numeric) and numeric >= 0.0:
                    self._metrics[cohort][stage].append(numeric)
            self._frames[cohort] += 1
            self._sources[str(source or "unknown")] += 1
            now = time.monotonic()
            if now - self._last_debug_at >= self.debug_interval_seconds:
                self._last_debug_at = now
                should_log = True
        if should_log:
            self._log_debug_summary()

    def record_drop(self, reason: str, count: int = 1) -> None:
        if not self.enabled or count <= 0:
            return
        with self._lock:
            self._drops[str(reason or "unknown")] += int(count)

    def _log_debug_summary(self) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        snapshot = self.snapshot()
        for cohort in (COHORT_WITH_PERSON, COHORT_WITHOUT_PERSON):
            populated = {
                stage: values
                for stage, values in snapshot["cohorts"][cohort].items()
                if values["samples"] > 0
            }
            logger.debug(
                "Pipeline performance cohort=%s frames=%s stages=%s",
                cohort,
                snapshot["frame_counts"][cohort],
                populated,
            )
        logger.debug(
            "Pipeline dropped frames: %s",
            snapshot["dropped_frames"],
        )

    def snapshot(self) -> dict:
        with self._lock:
            copied = {
                cohort: {
                    stage: list(values)
                    for stage, values in stages.items()
                }
                for cohort, stages in self._metrics.items()
            }
            frames = dict(self._frames)
            sources = dict(self._sources)
            drops = dict(self._drops)
            started_at = self._started_at

        cohorts = {
            cohort: {
                stage: _summary(copied[cohort][stage])
                for stage in PERFORMANCE_STAGES
            }
            for cohort in PERFORMANCE_COHORTS
        }
        combined = {
            stage: _summary(
                copied[COHORT_WITH_PERSON][stage]
                + copied[COHORT_WITHOUT_PERSON][stage]
                + copied[COHORT_UNKNOWN][stage]
            )
            for stage in PERFORMANCE_STAGES
        }
        return {
            "enabled": self.enabled,
            "window_size": self.window_size,
            "started_at": started_at,
            "generated_at": time.time(),
            "frame_counts": {
                cohort: int(frames.get(cohort, 0))
                for cohort in PERFORMANCE_COHORTS
            },
            "sources": sources,
            "dropped_frames": {
                "total": int(sum(drops.values())),
                "by_reason": drops,
                "cohort_note": (
                    "A replaced/rejected frame is dropped before inference, "
                    "so person/no-person classification is unavailable."
                ),
            },
            "cohorts": cohorts,
            "all": combined,
        }

    def reset(self) -> None:
        with self._lock:
            for stages in self._metrics.values():
                for values in stages.values():
                    values.clear()
            self._frames.clear()
            self._sources.clear()
            self._drops.clear()
            self._started_at = time.time()
            self._last_debug_at = time.monotonic()


__all__ = [
    "COHORT_UNKNOWN",
    "COHORT_WITHOUT_PERSON",
    "COHORT_WITH_PERSON",
    "FramePerformanceTrace",
    "PERFORMANCE_STAGES",
    "PerformanceProfiler",
]
