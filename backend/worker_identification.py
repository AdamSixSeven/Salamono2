"""Lightweight worker identification using visible worker tags.

The module intentionally does not use face recognition. A worker can wear a
simplified ArUco-style marker on the vest/back. The tag is
matched to the enclosing YOLO person box and cached briefly across occlusions.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import math
import threading
import time
from typing import Protocol

import cv2
import numpy as np

from backend.detector import Detection
from backend.worker_tags import worker_marker_id
from backend.worker_store import WorkerStore
from backend.danger_rules import bbox_iou
from config import CONFIG, WorkerIDConfig


logger = logging.getLogger(__name__)


@dataclass
class DecodedWorkerTag:
    payload: str
    polygon: list[tuple[float, float]]
    source: str = "marker"

    @property
    def center(self) -> tuple[float, float]:
        if not self.polygon:
            return (0.0, 0.0)
        return (
            sum(point[0] for point in self.polygon) / len(self.polygon),
            sum(point[1] for point in self.polygon) / len(self.polygon),
        )


@dataclass
class WorkerIdentity:
    worker_id: str
    person: Detection
    source: str
    tag_polygon: list[tuple[float, float]]
    frame_timestamp: float
    cached: bool = False
    track_id: int | None = None
    payload: str | None = None
    confidence: float | None = None
    marker_id: int | None = None
    identity_status: str = "confirmed"
    marker_age_sec: float = 0.0
    alert_eligible: bool = True


@dataclass
class _CacheEntry:
    worker_id: str
    person_box: tuple[int, int, int, int]
    tag_polygon: list[tuple[float, float]]
    last_seen: float


class WorkerTagDecoder(Protocol):
    def decode(self, frame_bgr: np.ndarray) -> list[DecodedWorkerTag]: ...


class OpenCVWorkerMarkerDecoder:
    """Decode only the simplified worker marker.

    Legacy QR decoding was intentionally removed. Worker identification now
    uses one compact ArUco 4x4 marker path, which is both faster and more
    stable at distance.
    """

    def __init__(self):
        self._dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, "DICT_4X4_1000", cv2.aruco.DICT_4X4_250)
        )
        self._params = cv2.aruco.DetectorParameters()
        self._detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector = cv2.aruco.ArucoDetector(
                self._dictionary,
                self._params,
            )

    def decode(self, frame_bgr: np.ndarray) -> list[DecodedWorkerTag]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray,
                self._dictionary,
                parameters=self._params,
            )
        if ids is None or len(ids) == 0:
            return []

        tags: list[DecodedWorkerTag] = []
        for index, marker_id in enumerate(ids.flatten().tolist()):
            points = corners[index].reshape(-1, 2)
            tags.append(DecodedWorkerTag(
                payload=f"marker:{int(marker_id)}",
                polygon=[(float(x), float(y)) for x, y in points],
                source="marker",
            ))
        return tags

def _center_distance_ratio(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    ax = (box_a[0] + box_a[2]) / 2.0
    ay = (box_a[1] + box_a[3]) / 2.0
    bx = (box_b[0] + box_b[2]) / 2.0
    by = (box_b[1] + box_b[3]) / 2.0
    scale = max(1.0, math.hypot(box_b[2] - box_b[0], box_b[3] - box_b[1]))
    return math.hypot(ax - bx, ay - by) / scale


def _translated_polygon(
    polygon: list[tuple[float, float]],
    old_box: tuple[int, int, int, int],
    new_box: tuple[int, int, int, int],
) -> list[tuple[float, float]]:
    """Move a cached tag with its person without scaling the marker shape."""
    old_cx = (old_box[0] + old_box[2]) / 2.0
    old_cy = (old_box[1] + old_box[3]) / 2.0
    new_cx = (new_box[0] + new_box[2]) / 2.0
    new_cy = (new_box[1] + new_box[3]) / 2.0
    dx = new_cx - old_cx
    dy = new_cy - old_cy
    return [(x + dx, y + dy) for x, y in polygon]


class WorkerIdentifier:
    def __init__(
        self,
        config: WorkerIDConfig | None = None,
        decoder: WorkerTagDecoder | None = None,
    ):
        self.cfg = config or CONFIG.worker_id
        self.decoder = decoder or OpenCVWorkerMarkerDecoder()
        self.available = bool(self.cfg.enabled)
        self.unavailable_reason = None if self.available else "disabled by WORKER_ID_ENABLED"
        self._last_sample_at: dict[str, float] = {}
        self._cache: dict[str, list[_CacheEntry]] = {}
        self._worker_store: WorkerStore | None = None
        self._marker_to_worker_id: dict[int, str] = {}
        self._marker_index_refreshed_at: float = float("-inf")
        self._marker_index_lock = threading.RLock()

    def _refresh_marker_index(self) -> None:
        """Refresh marker-to-worker mappings lazily on the worker thread."""
        current = time.monotonic()
        with self._marker_index_lock:
            if current - self._marker_index_refreshed_at < 5.0:
                return
            self._marker_index_refreshed_at = current
            try:
                self._worker_store = self._worker_store or WorkerStore(
                    self.cfg.database_path,
                )
                grouped: dict[int, list[str]] = {}
                for record in self._worker_store.list():
                    grouped.setdefault(worker_marker_id(record.worker_id), []).append(record.worker_id)
                conflicts = {mid: ids for mid, ids in grouped.items() if len(ids) != 1}
                for marker_id, worker_ids in conflicts.items():
                    logger.warning(
                        "Conflicting worker registry marker_id=%s workers=%s; marker ignored",
                        marker_id, ", ".join(worker_ids),
                    )
                self._marker_to_worker_id = {
                    marker_id: worker_ids[0]
                    for marker_id, worker_ids in grouped.items()
                    if len(worker_ids) == 1
                }
            except Exception:
                logger.warning(
                    "Worker marker index refresh failed; retaining cached map",
                    exc_info=True,
                )

    def invalidate_marker_index(self) -> None:
        """Force the next ArUco result to observe successful profile CRUD."""
        with self._marker_index_lock:
            self._marker_to_worker_id.clear()
            self._marker_index_refreshed_at = float("-inf")

    def _worker_id(self, payload: str) -> str | None:
        payload = payload.strip()
        if not payload.startswith("marker:"):
            return None
        raw = payload.split(":", 1)[1].strip()
        if not raw.isdigit():
            return None
        marker_id = int(raw)
        self._refresh_marker_index()
        return self._marker_to_worker_id.get(marker_id)

    def process(
        self,
        camera_id: str,
        frame_bgr: np.ndarray,
        persons: list[Detection],
        timestamp: float,
    ) -> list[WorkerIdentity]:
        if not self.available or not persons:
            return []

        self._expire(camera_id, timestamp)
        interval = 1.0 / max(self.cfg.sample_fps, 0.1)
        should_sample = timestamp - self._last_sample_at.get(camera_id, float("-inf")) >= interval
        tags: list[DecodedWorkerTag] = []
        if should_sample:
            self._last_sample_at[camera_id] = timestamp
            tags = self.decoder.decode(frame_bgr)

        identities = self._match_live_tags(camera_id, persons, tags, timestamp)
        matched_boxes = {identity.person.box for identity in identities}
        used_workers = {identity.worker_id for identity in identities}

        # Reuse a recent tag assignment through short occlusions or motion blur.
        cache = self._cache.get(camera_id, [])
        candidates: list[tuple[float, int, int]] = []
        for person_idx, person in enumerate(persons):
            if person.box in matched_boxes:
                continue
            for cache_idx, entry in enumerate(cache):
                if entry.worker_id in used_workers:
                    continue
                iou = bbox_iou(person.box, entry.person_box)
                dist = _center_distance_ratio(person.box, entry.person_box)
                if iou >= 0.15 or dist <= 0.55:
                    score = iou + 0.35 * max(0.0, 1.0 - dist)
                    candidates.append((score, person_idx, cache_idx))

        used_people: set[int] = set()
        used_cache: set[int] = set()
        for _, person_idx, cache_idx in sorted(candidates, reverse=True):
            if person_idx in used_people or cache_idx in used_cache:
                continue
            entry = cache[cache_idx]
            if entry.worker_id in used_workers:
                continue
            person = persons[person_idx]
            entry.tag_polygon = _translated_polygon(
                entry.tag_polygon,
                entry.person_box,
                person.box,
            )
            entry.person_box = person.box
            identities.append(WorkerIdentity(
                worker_id=entry.worker_id,
                person=person,
                source="cache",
                tag_polygon=list(entry.tag_polygon),
                frame_timestamp=timestamp,
                cached=True,
            ))
            used_people.add(person_idx)
            used_cache.add(cache_idx)
            used_workers.add(entry.worker_id)
        return identities

    def _match_live_tags(
        self,
        camera_id: str,
        persons: list[Detection],
        tags: list[DecodedWorkerTag],
        timestamp: float,
    ) -> list[WorkerIdentity]:
        parsed: list[tuple[str, DecodedWorkerTag]] = []
        for tag in tags:
            worker_id = self._worker_id(tag.payload)
            if worker_id is not None:
                parsed.append((worker_id, tag))

        candidates: list[tuple[float, int, int]] = []
        for tag_idx, (_, tag) in enumerate(parsed):
            cx, cy = tag.center
            for person_idx, person in enumerate(persons):
                x1, y1, x2, y2 = person.box
                pad_x = (x2 - x1) * self.cfg.match_padding
                pad_y = (y2 - y1) * self.cfg.match_padding
                if not (x1 - pad_x <= cx <= x2 + pad_x and y1 - pad_y <= cy <= y2 + pad_y):
                    continue
                pcx, pcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                scale = max(1.0, math.hypot(x2 - x1, y2 - y1))
                distance = math.hypot(cx - pcx, cy - pcy) / scale
                candidates.append((1.0 - distance, person_idx, tag_idx))

        identities: list[WorkerIdentity] = []
        used_people: set[int] = set()
        used_tags: set[int] = set()
        used_workers: set[str] = set()
        for _, person_idx, tag_idx in sorted(candidates, reverse=True):
            worker_id, tag = parsed[tag_idx]
            if person_idx in used_people or tag_idx in used_tags or worker_id in used_workers:
                continue
            person = persons[person_idx]
            identities.append(WorkerIdentity(
                worker_id=worker_id,
                person=person,
                source=tag.source,
                tag_polygon=list(tag.polygon),
                frame_timestamp=timestamp,
                cached=False,
            ))
            self._upsert_cache(camera_id, worker_id, person.box, tag.polygon, timestamp)
            used_people.add(person_idx)
            used_tags.add(tag_idx)
            used_workers.add(worker_id)
        return identities

    def _upsert_cache(
        self,
        camera_id: str,
        worker_id: str,
        person_box: tuple[int, int, int, int],
        polygon: list[tuple[float, float]],
        timestamp: float,
    ) -> None:
        entries = self._cache.setdefault(camera_id, [])
        for entry in entries:
            if entry.worker_id == worker_id:
                entry.person_box = person_box
                entry.tag_polygon = list(polygon)
                entry.last_seen = timestamp
                return
        entries.append(_CacheEntry(worker_id, person_box, list(polygon), timestamp))

    def _expire(self, camera_id: str, timestamp: float) -> None:
        entries = self._cache.get(camera_id, [])
        self._cache[camera_id] = [
            entry for entry in entries
            if timestamp - entry.last_seen <= self.cfg.cache_ttl_seconds
        ]


@dataclass(frozen=True, slots=True)
class WorkerIDWorkerStats:
    """Small, lock-consistent diagnostic snapshot for readiness/debugging."""

    scan_ms: float = 0.0
    queue_age_ms: float = 0.0
    crop_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    dropped_jobs: int = 0
    pending_jobs: int = 0
    errors: int = 0
    thread_alive: bool = False


@dataclass(frozen=True, slots=True)
class WorkerIDWorkerSnapshot:
    """One completed background scan.

    ``identities`` belongs to the source frame and is useful for diagnostics.
    Live overlays should call :meth:`WorkerIDWorker.current_identities`, which
    binds the cached label to the newest person box and intentionally omits
    historical marker corners.
    """

    camera_id: str
    frame_timestamp: float
    completed_at: float
    identities: tuple[WorkerIdentity, ...]
    scanned_track_ids: frozenset[int]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerMarkerScan:
    """Immutable evidence tying markers and person boxes to one source frame."""

    camera_id: str
    frame_id: int
    frame_timestamp: float
    captured_monotonic: float
    tags: tuple[DecodedWorkerTag, ...]
    persons: tuple[Detection, ...]


@dataclass(slots=True)
class _WorkerPersonTrack:
    box: tuple[int, int, int, int]
    last_seen_monotonic: float


@dataclass(slots=True)
class IdentityState:
    worker_id: str | None = None
    marker_id: int | None = None
    status: str = "unassigned"
    confidence: float = 0.0
    confirmed_at: float | None = None
    last_marker_seen_at: float | None = None
    last_track_seen_at: float | None = None
    candidate_marker_id: int | None = None
    candidate_worker_id: str | None = None
    candidate_matches: int = 0
    switch_candidate_marker_id: int | None = None
    switch_candidate_worker_id: str | None = None
    switch_candidate_matches: int = 0
    ambiguity_started_at: float | None = None
    hold_started_at: float | None = None
    person_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    marker_corners: list[tuple[float, float]] = None
    last_scan_frame_id: int = -1

    def __post_init__(self) -> None:
        if self.marker_corners is None:
            self.marker_corners = []


@dataclass(frozen=True, slots=True)
class _WorkerIDJob:
    camera_id: str
    frame: np.ndarray
    persons: tuple[Detection, ...]
    frame_timestamp: float
    queued_at_monotonic: float
    generation: int


class WorkerIDWorker:
    """Asynchronous full-frame worker identification with a one-slot queue.

    Marker results and person tracks are captured from the same frame. Identity
    state remains isolated by ``(camera_id, track_id)``.
    """

    def __init__(
        self,
        identifier: WorkerIdentifier,
        config: WorkerIDConfig | None = None,
        *,
        sample_fps: float | None = None,
        cache_ttl_seconds: float | None = None,
        crop_padding: float | None = None,
        full_frame_fallback: bool | None = None,
        max_pending_frames: int | None = None,
        autostart: bool = True,
        thread_name: str = "worker-id-worker",
        monotonic_clock=None,
        wall_clock=None,
    ):
        if not isinstance(identifier, WorkerIdentifier):
            raise TypeError("identifier must be a WorkerIdentifier")
        cfg = config or identifier.cfg
        self.identifier = identifier
        self.cfg = cfg
        self.sample_fps = max(
            0.0,
            float(
                sample_fps
                if sample_fps is not None
                else getattr(cfg, "sample_fps", 1.5)
            ),
        )
        self.cache_ttl_seconds = max(
            0.0,
            float(
                cache_ttl_seconds
                if cache_ttl_seconds is not None
                else getattr(cfg, "cache_ttl_seconds", 4.0)
            ),
        )
        self.display_ttl_seconds = max(0.0, float(cfg.display_ttl_seconds))
        self.crop_padding = max(
            0.0,
            float(
                crop_padding
                if crop_padding is not None
                else getattr(cfg, "crop_padding", 0.15)
            ),
        )
        self.full_frame_fallback = bool(
            full_frame_fallback
            if full_frame_fallback is not None
            else getattr(cfg, "full_frame_fallback", False)
        )
        pending_limit = int(
            max_pending_frames
            if max_pending_frames is not None
            else getattr(cfg, "max_pending_frames", 1)
        )
        if pending_limit != 1:
            logger.warning(
                "WORKER_ID_MAX_PENDING_FRAMES=%s is unsupported; clamping to 1",
                pending_limit,
            )
        self.max_pending_frames = 1

        self.available = bool(identifier.available)
        self.thread_name = str(thread_name)
        self._monotonic = monotonic_clock or time.monotonic
        self._wall_clock = wall_clock or time.time
        self._condition = threading.Condition()
        self._pending: _WorkerIDJob | None = None
        self._active = False
        self._closed = False
        self._disabled_cameras: set[str] = set()
        self._thread: threading.Thread | None = None

        self._tracks: dict[str, dict[int, _WorkerPersonTrack]] = {}
        self._next_track_id: dict[str, int] = {}
        self._active_track_ids: dict[str, set[int]] = {}
        self._identity_cache: dict[tuple[str, int], IdentityState] = {}
        self._scanned_tracks: dict[str, set[int]] = {}
        self._last_scan_started_at: dict[str, float] = {}
        self._results: dict[str, WorkerIDWorkerSnapshot] = {}
        self._camera_generations: dict[str, int] = {}
        self._frame_ids: dict[str, int] = {}

        self._scan_ms = 0.0
        self._queue_age_ms = 0.0
        self._crop_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._dropped_jobs = 0
        self._errors = 0
        if autostart:
            self.start()

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("WorkerIDWorker is closed")
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run,
                name=self.thread_name,
                daemon=True,
            )
            self._thread.start()

    def submit_latest(
        self,
        camera_id: str,
        frame: np.ndarray,
        persons: list[Detection],
        timestamp: float,
        *,
        _copy_frame: bool = True,
    ) -> list[Detection]:
        """Track people and enqueue only the newest due marker scan."""

        camera_id = str(camera_id).strip()
        if not camera_id:
            raise ValueError("camera_id must not be empty")
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array")
        now = float(self._monotonic())
        people = list(persons)
        with self._condition:
            generation = self._camera_generations.get(camera_id, 0)
            previous_active_ids = set(self._active_track_ids.get(camera_id, set()))
            tracked_people = self._assign_track_ids_locked(
                camera_id,
                people,
                now,
            )
            active_ids = {
                int(person.track_id)
                for person in tracked_people
                if person.track_id is not None
            }
            self._reacquire_identity_locked(
                camera_id, tracked_people, previous_active_ids, active_ids, now,
            )
            self._active_track_ids[camera_id] = active_ids
            for person in tracked_people:
                if person.track_id is None:
                    continue
                state = self._identity_cache.get((camera_id, int(person.track_id)))
                if state is not None:
                    if (
                        state.worker_id
                        and _center_distance_ratio(person.box, state.person_box) > 0.65
                    ):
                        state.status = "ambiguous"
                        state.ambiguity_started_at = state.ambiguity_started_at or now
                        logger.debug(
                            "Worker identity ambiguous camera=%s track=%s reason=unrealistic_jump",
                            camera_id, person.track_id,
                        )
                    state.last_track_seen_at = now
                    state.person_box = tuple(person.box)
            self._drop_missing_tracks_locked(camera_id, active_ids)
            self._mark_overlaps_ambiguous_locked(camera_id, tracked_people, now)
            self._expire_cache_locked(now)

            if (
                self._closed
                or not self.available
                or camera_id in self._disabled_cameras
            ):
                return tracked_people
            if not tracked_people and not self.full_frame_fallback:
                if self._pending is not None and self._pending.camera_id == camera_id:
                    self._pending = None
                    self._dropped_jobs += 1
                self._condition.notify_all()
                return tracked_people
            interval = (
                float("inf") if self.sample_fps <= 0.0
                else 1.0 / self.sample_fps
            )
            last_started = self._last_scan_started_at.get(
                camera_id,
                float("-inf"),
            )
            if now - last_started < interval:
                return tracked_people

        # Copy outside the condition so readers of cached results remain fast.
        frame_copy = (
            np.array(frame, copy=True, order="C")
            if _copy_frame else frame
        )
        people_copy = tuple(replace(person) for person in tracked_people)
        job = _WorkerIDJob(
            camera_id=camera_id,
            frame=frame_copy,
            persons=people_copy,
            frame_timestamp=float(timestamp),
            queued_at_monotonic=now,
            generation=generation,
        )
        with self._condition:
            if (
                self._closed
                or camera_id in self._disabled_cameras
                or not self.available
                or generation != self._camera_generations.get(camera_id, 0)
            ):
                return tracked_people
            if self._pending is not None:
                self._dropped_jobs += 1
            self._pending = job
            self._condition.notify()
        return tracked_people

    def submit_borrowed_latest(
        self,
        camera_id: str,
        frame: np.ndarray,
        persons: list[Detection],
        timestamp: float,
    ) -> list[Detection]:
        """Internal zero-copy submission for an immutable decoded FrameJob."""

        return self.submit_latest(
            camera_id,
            frame,
            persons,
            timestamp,
            _copy_frame=False,
        )

    def current_identities(
        self,
        camera_id: str,
        persons: list[Detection],
        timestamp: float,
    ) -> list[WorkerIdentity]:
        """Materialize valid IDs on current boxes without historical corners."""

        camera_id = str(camera_id)
        now = float(self._monotonic())
        with self._condition:
            self._expire_cache_locked(now)
            self._mark_overlaps_ambiguous_locked(camera_id, persons, now)
            identities: list[WorkerIdentity] = []
            for person in persons:
                if person.track_id is None:
                    self._cache_misses += 1
                    continue
                track_id = int(person.track_id)
                if track_id not in self._active_track_ids.get(camera_id, set()):
                    self._cache_misses += 1
                    continue
                entry = self._identity_cache.get((camera_id, track_id))
                if entry is None or not entry.worker_id or entry.status in {
                    "unassigned", "candidate", "switch_pending", "ambiguous", "expired",
                }:
                    self._cache_misses += 1
                    continue
                profile = self.identifier._worker_store.get(entry.worker_id) if self.identifier._worker_store else None
                self.identifier._refresh_marker_index()
                marker_owner = self.identifier._marker_to_worker_id.get(entry.marker_id)
                if profile is None or marker_owner != entry.worker_id:
                    self._identity_cache.pop((camera_id, track_id), None)
                    self._cache_misses += 1
                    continue
                marker_age = (
                    float("inf") if entry.last_marker_seen_at is None
                    else max(0.0, now - entry.last_marker_seen_at)
                )
                if marker_age > self.display_ttl_seconds:
                    entry.status = "expired"
                    self._cache_misses += 1
                    continue
                self._cache_hits += 1
                identities.append(WorkerIdentity(
                    worker_id=entry.worker_id,
                    person=person,
                    source="cache",
                    tag_polygon=[],
                    frame_timestamp=float(timestamp),
                    cached=True,
                    track_id=track_id,
                    payload=f"marker:{entry.marker_id}",
                    confidence=entry.confidence,
                    marker_id=entry.marker_id,
                    identity_status=entry.status,
                    marker_age_sec=marker_age,
                    alert_eligible=(
                        entry.status in {"confirmed", "held"}
                        and marker_age <= self.cfg.alert_max_age_seconds
                    ),
                ))
            return identities

    def identity_state(self, camera_id: str, track_id: int) -> IdentityState | None:
        """Return a defensive state snapshot for diagnostics/tests."""
        with self._condition:
            state = self._identity_cache.get((str(camera_id), int(track_id)))
            return replace(state) if state is not None else None

    def scanned_track_ids(self, camera_id: str) -> set[int]:
        """Return tracks for which at least one crop scan has completed."""

        with self._condition:
            active = self._active_track_ids.get(str(camera_id), set())
            scanned = self._scanned_tracks.get(str(camera_id), set())
            return set(active.intersection(scanned))

    def set_camera_enabled(self, camera_id: str, enabled: bool) -> None:
        camera_id = str(camera_id)
        with self._condition:
            if enabled:
                self._disabled_cameras.discard(camera_id)
            else:
                self._disabled_cameras.add(camera_id)
                self._camera_generations[camera_id] = (
                    self._camera_generations.get(camera_id, 0) + 1
                )
                if self._pending is not None and self._pending.camera_id == camera_id:
                    self._pending = None
                    self._dropped_jobs += 1
                self._clear_camera_locked(camera_id)
            self._condition.notify_all()

    def reset_camera(self, camera_id: str) -> None:
        """Discard one camera's tracks and invalidate an in-flight old run."""
        camera_id = str(camera_id)
        with self._condition:
            self._camera_generations[camera_id] = (
                self._camera_generations.get(camera_id, 0) + 1
            )
            if self._pending is not None and self._pending.camera_id == camera_id:
                self._pending = None
                self._dropped_jobs += 1
            self._clear_camera_locked(camera_id)
            self._condition.notify_all()

    def stats(self) -> WorkerIDWorkerStats:
        with self._condition:
            thread = self._thread
            return WorkerIDWorkerStats(
                scan_ms=self._scan_ms,
                queue_age_ms=self._queue_age_ms,
                crop_count=self._crop_count,
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
                dropped_jobs=self._dropped_jobs,
                pending_jobs=int(self._pending is not None),
                errors=self._errors,
                thread_alive=bool(thread and thread.is_alive()),
            )

    def wait_for_result(
        self,
        camera_id: str,
        *,
        after_timestamp: float | None = None,
        timeout: float = 2.0,
    ) -> WorkerIDWorkerSnapshot | None:
        deadline = time.monotonic() + max(0.0, float(timeout))
        camera_id = str(camera_id)
        with self._condition:
            while True:
                snapshot = self._results.get(camera_id)
                if snapshot is not None and (
                    after_timestamp is None
                    or snapshot.frame_timestamp > after_timestamp
                ):
                    return snapshot
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                if self._closed and not self._active and self._pending is None:
                    return None
                self._condition.wait(remaining)

    def close(
        self,
        *,
        wait: bool = True,
        timeout: float | None = 5.0,
    ) -> None:
        with self._condition:
            if not self._closed:
                self._closed = True
                if self._pending is not None:
                    self._pending = None
                    self._dropped_jobs += 1
                self._condition.notify_all()
            thread = self._thread
        if (
            wait
            and thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=timeout)

    @property
    def pending_count(self) -> int:
        with self._condition:
            return int(self._pending is not None)

    @property
    def busy(self) -> bool:
        with self._condition:
            return self._active or self._pending is not None

    def _assign_track_ids_locked(
        self,
        camera_id: str,
        persons: list[Detection],
        now: float,
    ) -> list[Detection]:
        tracks = self._tracks.setdefault(camera_id, {})
        track_ttl = max(self.cache_ttl_seconds, 1.0)
        for track_id in [
            track_id
            for track_id, track in tracks.items()
            if now - track.last_seen_monotonic > track_ttl
        ]:
            tracks.pop(track_id, None)

        used_tracks: set[int] = set()
        for person in persons:
            if person.track_id is None:
                continue
            track_id = int(person.track_id)
            person.track_id = track_id
            tracks[track_id] = _WorkerPersonTrack(
                box=tuple(int(value) for value in person.box),
                last_seen_monotonic=now,
            )
            used_tracks.add(track_id)
            self._next_track_id[camera_id] = max(
                self._next_track_id.get(camera_id, 1),
                track_id + 1,
            )

        candidates: list[tuple[float, int, int]] = []
        for person_index, person in enumerate(persons):
            if person.track_id is not None:
                continue
            for track_id, track in tracks.items():
                if track_id in used_tracks:
                    continue
                iou = bbox_iou(person.box, track.box)
                distance = _center_distance_ratio(person.box, track.box)
                if iou >= 0.15 or distance <= 0.55:
                    score = iou + 0.35 * max(0.0, 1.0 - distance)
                    candidates.append((score, person_index, track_id))

        matched_people: set[int] = set()
        for _, person_index, track_id in sorted(candidates, reverse=True):
            if person_index in matched_people or track_id in used_tracks:
                continue
            person = persons[person_index]
            person.track_id = track_id
            tracks[track_id] = _WorkerPersonTrack(
                box=tuple(int(value) for value in person.box),
                last_seen_monotonic=now,
            )
            matched_people.add(person_index)
            used_tracks.add(track_id)

        next_track_id = self._next_track_id.get(camera_id, 1)
        for person in persons:
            if person.track_id is not None:
                continue
            while next_track_id in tracks or next_track_id in used_tracks:
                next_track_id += 1
            person.track_id = next_track_id
            tracks[next_track_id] = _WorkerPersonTrack(
                box=tuple(int(value) for value in person.box),
                last_seen_monotonic=now,
            )
            used_tracks.add(next_track_id)
            next_track_id += 1
        self._next_track_id[camera_id] = next_track_id
        return persons

    def _drop_missing_tracks_locked(
        self,
        camera_id: str,
        active_ids: set[int],
    ) -> None:
        if camera_id in self._scanned_tracks:
            self._scanned_tracks[camera_id].intersection_update(active_ids)

    def _reacquire_identity_locked(
        self,
        camera_id: str,
        persons: list[Detection],
        previous_active_ids: set[int],
        active_ids: set[int],
        now: float,
    ) -> None:
        if not self.cfg.reacquire_enabled or len(persons) != 1:
            return
        person = persons[0]
        if person.track_id is None or int(person.track_id) in previous_active_ids:
            return
        new_track_id = int(person.track_id)
        candidates: list[tuple[tuple[str, int], IdentityState]] = []
        for key, state in self._identity_cache.items():
            if key[0] != camera_id or key[1] in active_ids or not state.worker_id:
                continue
            if state.status not in {"confirmed", "held"} or state.last_track_seen_at is None:
                continue
            if now - state.last_track_seen_at > self.cfg.reacquire_max_gap_seconds:
                continue
            old = state.person_box
            old_w, old_h = max(1, old[2] - old[0]), max(1, old[3] - old[1])
            new = person.box
            new_w, new_h = max(1, new[2] - new[0]), max(1, new[3] - new[1])
            if not (0.7 <= new_w / old_w <= 1.3 and 0.7 <= new_h / old_h <= 1.3):
                continue
            if _center_distance_ratio(new, old) > 0.20:
                continue
            candidates.append((key, state))
        if len(candidates) != 1:
            return
        old_key, state = candidates[0]
        if any(
            other_key[0] == camera_id and other_key != old_key
            and other.worker_id == state.worker_id
            and other.status in {"confirmed", "held"}
            for other_key, other in self._identity_cache.items()
        ):
            return
        self._identity_cache.pop(old_key, None)
        state.status = "held"
        state.hold_started_at = now
        state.last_track_seen_at = now
        state.person_box = tuple(person.box)
        self._identity_cache[(camera_id, new_track_id)] = state
        logger.debug("Worker identity handover camera=%s old_track=%s new_track=%s", camera_id, old_key[1], new_track_id)

    def _expire_cache_locked(self, now: float) -> None:
        for key in [
            key
            for key, entry in self._identity_cache.items()
            if (
                (
                    entry.last_marker_seen_at is not None
                    and now - entry.last_marker_seen_at > self.display_ttl_seconds
                )
                or (
                    entry.last_track_seen_at is not None
                    and now - entry.last_track_seen_at > max(self.display_ttl_seconds, 1.0)
                )
                or (
                    entry.last_marker_seen_at is None
                    and entry.last_track_seen_at is None
                )
            )
        ]:
            self._identity_cache[key].status = "expired"
            self._identity_cache.pop(key, None)

    def _mark_overlaps_ambiguous_locked(
        self, camera_id: str, persons: list[Detection], now: float,
    ) -> None:
        for index, first in enumerate(persons):
            if first.track_id is None:
                continue
            for second in persons[index + 1:]:
                if second.track_id is None:
                    continue
                if bbox_iou(first.box, second.box) < self.cfg.ambiguous_iou_threshold:
                    continue
                for track_id in (int(first.track_id), int(second.track_id)):
                    state = self._identity_cache.get((camera_id, track_id))
                    if state is not None and state.worker_id:
                        state.status = "ambiguous"
                        state.ambiguity_started_at = state.ambiguity_started_at or now
                        logger.debug("Worker identity ambiguous camera=%s track=%s reason=overlap", camera_id, track_id)

    def _clear_camera_locked(self, camera_id: str) -> None:
        self._tracks.pop(camera_id, None)
        self._next_track_id.pop(camera_id, None)
        self._active_track_ids.pop(camera_id, None)
        self._scanned_tracks.pop(camera_id, None)
        self._last_scan_started_at.pop(camera_id, None)
        self._results.pop(camera_id, None)
        self._frame_ids.pop(camera_id, None)
        for key in [
            key for key in self._identity_cache if key[0] == camera_id
        ]:
            self._identity_cache.pop(key, None)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                job = self._pending
                self._pending = None
                self._active = True
                scan_started = float(self._monotonic())
                self._last_scan_started_at[job.camera_id] = scan_started
                self._queue_age_ms = max(
                    0.0,
                    (scan_started - job.queued_at_monotonic) * 1000.0,
                )
                self._condition.notify_all()

            identities: list[WorkerIdentity] = []
            scanned_ids: set[int] = set()
            cache_updates: list[
                tuple[int, WorkerIdentity, DecodedWorkerTag]
            ] = []
            error: str | None = None
            crop_count = 0
            try:
                identities, scanned_ids, cache_updates, crop_count = (
                    self._scan_job(job)
                )
            except Exception as exc:  # keep the long-lived worker alive
                error = f"{type(exc).__name__}: {exc}"
                logger.exception(
                    "Worker ID scan failed for camera %s",
                    job.camera_id,
                )
            scan_finished = float(self._monotonic())

            with self._condition:
                self._scan_ms = max(
                    0.0,
                    (scan_finished - scan_started) * 1000.0,
                )
                self._crop_count = crop_count
                if error is not None:
                    self._errors += 1
                active_ids = self._active_track_ids.get(job.camera_id, set())
                enabled = (
                    not self._closed
                    and job.camera_id not in self._disabled_cameras
                    and job.generation
                    == self._camera_generations.get(job.camera_id, 0)
                )
                committed_scans = scanned_ids.intersection(active_ids)
                if enabled:
                    self._scanned_tracks.setdefault(
                        job.camera_id,
                        set(),
                    ).update(committed_scans)
                    committed_identities = self._apply_observations_locked(
                        job, cache_updates, committed_scans, scan_finished,
                    )
                    self._expire_cache_locked(scan_finished)
                    self._results[job.camera_id] = WorkerIDWorkerSnapshot(
                        camera_id=job.camera_id,
                        frame_timestamp=job.frame_timestamp,
                        completed_at=float(self._wall_clock()),
                        identities=tuple(committed_identities),
                        scanned_track_ids=frozenset(committed_scans),
                        error=error,
                    )
                self._active = False
                self._condition.notify_all()

    def _apply_observations_locked(
        self,
        job: _WorkerIDJob,
        observations: list[tuple[int, WorkerIdentity, DecodedWorkerTag]],
        scanned_ids: set[int],
        now: float,
    ) -> list[WorkerIdentity]:
        observed_tracks = {track_id for track_id, _, _ in observations}
        committed: list[WorkerIdentity] = []
        for track_id in scanned_ids - observed_tracks:
            state = self._identity_cache.get((job.camera_id, track_id))
            if state is not None and state.status == "confirmed":
                state.status = "held"
                state.hold_started_at = now
                logger.debug("Worker identity held camera=%s track=%s", job.camera_id, track_id)

        for track_id, identity, tag in observations:
            key = (job.camera_id, track_id)
            marker_id = int(tag.payload.split(":", 1)[1])
            state = self._identity_cache.setdefault(key, IdentityState())
            if state.last_scan_frame_id == self._frame_ids.get(job.camera_id, -1):
                continue
            state.last_scan_frame_id = self._frame_ids.get(job.camera_id, -1)
            state.last_track_seen_at = now
            state.person_box = tuple(identity.person.box)

            if state.worker_id == identity.worker_id and state.status in {"confirmed", "held", "switch_pending"}:
                if state.status == "switch_pending":
                    logger.debug("Worker identity switch cancelled camera=%s track=%s", job.camera_id, track_id)
                state.status = "confirmed"
                state.last_marker_seen_at = now
                state.confidence = float(identity.confidence or 1.0)
                state.marker_corners = list(tag.polygon)
                state.switch_candidate_marker_id = None
                state.switch_candidate_worker_id = None
                state.switch_candidate_matches = 0
                state.hold_started_at = None
                committed.append(replace(identity, marker_id=marker_id, identity_status="confirmed"))
                continue

            if state.worker_id and state.worker_id != identity.worker_id:
                if state.switch_candidate_marker_id == marker_id:
                    state.switch_candidate_matches += 1
                else:
                    state.switch_candidate_marker_id = marker_id
                    state.switch_candidate_worker_id = identity.worker_id
                    state.switch_candidate_matches = 1
                state.status = "switch_pending"
                logger.debug("Worker identity switch pending camera=%s track=%s matches=%s", job.camera_id, track_id, state.switch_candidate_matches)
                if state.switch_candidate_matches < self.cfg.change_confirmations:
                    continue
                logger.debug("Worker identity switch confirmed camera=%s track=%s worker=%s", job.camera_id, track_id, identity.worker_id)
            else:
                if state.candidate_marker_id == marker_id:
                    state.candidate_matches += 1
                else:
                    state.candidate_marker_id = marker_id
                    state.candidate_worker_id = identity.worker_id
                    state.candidate_matches = 1
                state.status = "candidate"
                logger.debug("Worker identity candidate camera=%s track=%s matches=%s", job.camera_id, track_id, state.candidate_matches)
                if state.candidate_matches < self.cfg.initial_confirmations:
                    continue

            if self.cfg.enforce_unique_active_id:
                conflicts = [
                    (other_key, other)
                    for other_key, other in self._identity_cache.items()
                    if other_key[0] == job.camera_id and other_key != key
                    and other.worker_id == identity.worker_id
                    and other.status in {"confirmed", "held"}
                ]
                fresh_conflict = False
                for other_key, other in conflicts:
                    age = float("inf") if other.last_marker_seen_at is None else now - other.last_marker_seen_at
                    if age <= self.cfg.alert_max_age_seconds and other.status == "confirmed":
                        other.status = "ambiguous"
                        state.status = "ambiguous"
                        fresh_conflict = True
                        logger.warning("Worker identity conflict camera=%s worker=%s tracks=%s,%s", job.camera_id, identity.worker_id, other_key[1], track_id)
                    else:
                        other.status = "expired"
                if fresh_conflict:
                    continue

            state.worker_id = identity.worker_id
            state.marker_id = marker_id
            state.status = "confirmed"
            state.confidence = float(identity.confidence or 1.0)
            state.confirmed_at = state.confirmed_at or now
            state.last_marker_seen_at = now
            state.marker_corners = list(tag.polygon)
            state.candidate_marker_id = None
            state.candidate_worker_id = None
            state.candidate_matches = 0
            state.switch_candidate_marker_id = None
            state.switch_candidate_worker_id = None
            state.switch_candidate_matches = 0
            state.ambiguity_started_at = None
            state.hold_started_at = None
            committed.append(replace(identity, marker_id=marker_id, identity_status="confirmed"))
            logger.debug("Worker marker confirmed camera=%s track=%s worker=%s", job.camera_id, track_id, identity.worker_id)
        return committed

    def _scan_job(
        self,
        job: _WorkerIDJob,
    ) -> tuple[
        list[WorkerIdentity],
        set[int],
        list[tuple[int, WorkerIdentity, DecodedWorkerTag]],
        int,
    ]:
        if not job.persons:
            return [], set(), [], 0

        # Exactly one detector call, on the untouched full-resolution source
        # frame. Person boxes in this job are the snapshot from that same frame.
        tags = self.identifier.decoder.decode(job.frame)
        scanned_ids = {
            int(person.track_id) for person in job.persons
            if person.track_id is not None
        }
        with self._condition:
            frame_id = self._frame_ids.get(job.camera_id, 0) + 1
            self._frame_ids[job.camera_id] = frame_id
        scan = WorkerMarkerScan(
            camera_id=job.camera_id,
            frame_id=frame_id,
            frame_timestamp=job.frame_timestamp,
            captured_monotonic=job.queued_at_monotonic,
            tags=tuple(tags),
            persons=job.persons,
        )

        candidates: list[tuple[float, int, Detection, str, DecodedWorkerTag]] = []
        rejected_unknown = 0
        ambiguous_tags: set[int] = set()
        ambiguous_track_ids: set[int] = set()
        for index, first in enumerate(scan.persons):
            if first.track_id is None:
                continue
            for second in scan.persons[index + 1:]:
                if second.track_id is None:
                    continue
                if bbox_iou(first.box, second.box) >= self.cfg.ambiguous_iou_threshold:
                    ambiguous_track_ids.update((int(first.track_id), int(second.track_id)))
        for tag_index, tag in enumerate(scan.tags):
            worker_id = self.identifier._worker_id(tag.payload)
            if worker_id is None:
                rejected_unknown += 1
                continue
            tag_cx, tag_cy = tag.center
            matches: list[tuple[float, int, Detection]] = []
            for person in scan.persons:
                if person.track_id is None:
                    continue
                if int(person.track_id) in ambiguous_track_ids:
                    continue
                x1, y1, x2, y2 = person.box
                width, height = x2 - x1, y2 - y1
                pad_x = width * min(0.10, self.crop_padding)
                pad_y = height * min(0.10, self.crop_padding)
                if not (x1 - pad_x <= tag_cx <= x2 + pad_x and y1 - pad_y <= tag_cy <= y2 + pad_y):
                    continue
                torso_y1, torso_y2 = y1 + 0.15 * height, y1 + 0.75 * height
                torso_cx, torso_cy = (x1 + x2) / 2.0, (torso_y1 + torso_y2) / 2.0
                distance = math.hypot(tag_cx - torso_cx, tag_cy - torso_cy) / max(1.0, math.hypot(width, height))
                torso_bonus = 0.35 if torso_y1 <= tag_cy <= torso_y2 else 0.0
                matches.append((torso_bonus + 1.0 - distance, int(person.track_id), person))
            matches.sort(key=lambda item: item[0], reverse=True)
            # Fail closed when two overlapping boxes score nearly the same.
            if len(matches) > 1 and matches[0][0] - matches[1][0] < 0.12:
                ambiguous_tags.add(tag_index)
                continue
            if matches:
                score, track_id, person = matches[0]
                candidates.append((score, track_id, person, worker_id, tag))

        identities: list[WorkerIdentity] = []
        updates: list[tuple[int, WorkerIdentity, DecodedWorkerTag]] = []
        used_tracks: set[int] = set()
        used_workers: set[str] = set()
        for score, track_id, person, worker_id, tag in sorted(
            candidates,
            key=lambda candidate: candidate[0],
            reverse=True,
        ):
            if track_id in used_tracks or worker_id in used_workers:
                continue
            confidence = max(0.0, min(1.0, float(score)))
            marker_id = int(tag.payload.split(":", 1)[1])
            identity = WorkerIdentity(
                worker_id=worker_id,
                person=person,
                source=tag.source,
                tag_polygon=list(tag.polygon),
                frame_timestamp=job.frame_timestamp,
                cached=False,
                track_id=track_id,
                payload=tag.payload,
                confidence=confidence,
            )
            identities.append(identity)
            updates.append((track_id, identity, tag))
            used_tracks.add(track_id)
            used_workers.add(worker_id)
        if getattr(self.cfg, "diagnostic_logging", False):
            logger.info(
                "Worker ArUco full-frame scan camera=%s frame=%s markers=%s unknown=%s ambiguous=%s assignments=%s",
                job.camera_id, frame_id, len(tags), rejected_unknown,
                len(ambiguous_tags),
                [(item.track_id, item.payload, item.worker_id) for item in identities],
            )
        return identities, scanned_ids, updates, 1


@dataclass
class UnidentifiedWorkerEvent:
    person: Detection
    frame_timestamp: float
    severity: str = "WARNING"
    confirmed: bool = True


class UnidentifiedWorkerMonitor:
    """Temporal warning for people without a visible/cached worker marker.

    This is intentionally enabled by policy per mode.  The checkpoint default
    is useful because the person is expected to present the tag; site-wide use
    remains opt-in due to occlusions and workers facing away from the camera.
    """

    def __init__(self, config: WorkerIDConfig | None = None):
        from collections import defaultdict
        self.cfg = config or CONFIG.worker_id
        self._streak = defaultdict(int)
        self._last_alert: dict[str, float] = {}

    @staticmethod
    def _cell(person: Detection) -> tuple[int, int]:
        x1, y1, x2, y2 = person.box
        return ((x1 + x2) // 96, (y1 + y2) // 96)

    def update(
        self,
        camera_id: str,
        mode: str,
        persons: list[Detection],
        identities: list[WorkerIdentity],
        timestamp: float,
    ) -> list[UnidentifiedWorkerEvent]:
        required = (
            self.cfg.require_at_checkpoint if mode == "checkpoint"
            else self.cfg.require_on_site
        )
        if not required:
            return []
        identified_tracks = {
            int(identity.track_id)
            for identity in identities
            if identity.track_id is not None
        }
        identified_boxes = {tuple(identity.person.box) for identity in identities}
        current: set[str] = set()
        confirmed: list[UnidentifiedWorkerEvent] = []
        for person in persons:
            if (
                person.track_id is not None
                and int(person.track_id) in identified_tracks
            ) or tuple(person.box) in identified_boxes:
                continue
            person_key = (
                f"track:{int(person.track_id)}"
                if person.track_id is not None
                else f"cell:{self._cell(person)}"
            )
            key = f"{camera_id}:{mode}:{person_key}"
            current.add(key)
            self._streak[key] += 1
            if self._streak[key] < max(1, self.cfg.unidentified_frames_required):
                continue
            last = self._last_alert.get(key, float("-inf"))
            if timestamp - last < self.cfg.unidentified_cooldown_seconds:
                continue
            confirmed.append(UnidentifiedWorkerEvent(person=person, frame_timestamp=timestamp))
            self._last_alert[key] = timestamp
        stale = [key for key in self._streak if key.startswith(f"{camera_id}:{mode}:") and key not in current]
        for key in stale:
            self._streak.pop(key, None)
        return confirmed

    def reset_camera(self, camera_id: str) -> None:
        prefix = f"{camera_id}:"
        for mapping in (self._streak, self._last_alert):
            for key in [key for key in mapping if key.startswith(prefix)]:
                mapping.pop(key, None)
