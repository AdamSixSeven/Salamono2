"""Lightweight worker identification using visible worker tags.

The module intentionally does not use face recognition. A worker can wear a
simplified ArUco-style marker on the vest/back. The tag is
matched to the enclosing YOLO person box and cached briefly across occlusions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import cv2
import numpy as np

from backend.detector import Detection
from backend.worker_tags import worker_id_from_marker_id, worker_marker_id
from backend.worker_store import WorkerStore
from backend.danger_rules import bbox_iou
from config import CONFIG, WorkerIDConfig


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

    def _refresh_marker_index(self) -> None:
        # Refresh lazily no more often than once every 5 seconds.
        import time
        current = time.time()
        if current - self._marker_index_refreshed_at < 5.0 and self._marker_to_worker_id:
            return
        self._marker_index_refreshed_at = current
        try:
            self._worker_store = self._worker_store or WorkerStore(self.cfg.database_path)
            mapping: dict[int, str] = {}
            for record in self._worker_store.list():
                mapping[worker_marker_id(record.worker_id)] = record.worker_id
            self._marker_to_worker_id = mapping
        except Exception:
            # Keep the detector running even when the worker database is not yet available.
            self._marker_to_worker_id = self._marker_to_worker_id or {}

    def _worker_id(self, payload: str) -> str | None:
        payload = payload.strip()
        if not payload.startswith("marker:"):
            return None
        raw = payload.split(":", 1)[1].strip()
        if not raw.isdigit():
            return None
        marker_id = int(raw)
        self._refresh_marker_index()
        return self._marker_to_worker_id.get(
            marker_id,
            worker_id_from_marker_id(marker_id),
        )

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
        identified_boxes = {tuple(identity.person.box) for identity in identities}
        current: set[str] = set()
        confirmed: list[UnidentifiedWorkerEvent] = []
        for person in persons:
            if tuple(person.box) in identified_boxes:
                continue
            key = f"{camera_id}:{mode}:{self._cell(person)}"
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
