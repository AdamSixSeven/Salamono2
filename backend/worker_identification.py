"""Lightweight worker identification using visible QR tags.

The module intentionally does not use face recognition.  A worker can wear a
QR tag on the vest/back with payload ``worker:<identifier>``.  The tag is
matched to the enclosing YOLO person box and cached briefly across occlusions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import cv2
import numpy as np

from backend.detector import Detection
from backend.danger_rules import bbox_iou
from config import CONFIG, WorkerIDConfig


@dataclass
class DecodedWorkerTag:
    payload: str
    polygon: list[tuple[float, float]]

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


class OpenCVQRDecoder:
    def __init__(self):
        self._detector = cv2.QRCodeDetector()

    def decode(self, frame_bgr: np.ndarray) -> list[DecodedWorkerTag]:
        tags: list[DecodedWorkerTag] = []
        try:
            result = self._detector.detectAndDecodeMulti(frame_bgr)
        except cv2.error:
            result = None

        if result:
            # OpenCV returns (retval, decoded_info, points, straight_qrcode).
            ok, decoded_info, points, *_ = result
            if ok and points is not None:
                for payload, polygon in zip(decoded_info, points):
                    if not payload:
                        continue
                    pts = [(float(x), float(y)) for x, y in np.asarray(polygon).reshape(-1, 2)]
                    tags.append(DecodedWorkerTag(str(payload), pts))

        if tags:
            return tags

        # Fallback for OpenCV builds where multi decode is unavailable or
        # returns false for a single code.
        try:
            payload, points, _ = self._detector.detectAndDecode(frame_bgr)
        except cv2.error:
            return []
        if payload and points is not None:
            pts = [(float(x), float(y)) for x, y in np.asarray(points).reshape(-1, 2)]
            tags.append(DecodedWorkerTag(str(payload), pts))
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


class WorkerIdentifier:
    def __init__(
        self,
        config: WorkerIDConfig | None = None,
        decoder: WorkerTagDecoder | None = None,
    ):
        self.cfg = config or CONFIG.worker_id
        self.decoder = decoder or OpenCVQRDecoder()
        self.available = bool(self.cfg.enabled)
        self.unavailable_reason = None if self.available else "disabled by WORKER_ID_ENABLED"
        self._last_sample_at: dict[str, float] = {}
        self._cache: dict[str, list[_CacheEntry]] = {}

    def _worker_id(self, payload: str) -> str | None:
        payload = payload.strip()
        if not payload or len(payload) > self.cfg.max_payload_length:
            return None
        if self.cfg.prefix and not payload.startswith(self.cfg.prefix):
            return None
        worker_id = payload[len(self.cfg.prefix):].strip() if self.cfg.prefix else payload
        if not worker_id or len(worker_id) > 64:
            return None
        # Keep IDs printable and safe for CSV/UI rendering.
        if any(ord(char) < 32 for char in worker_id):
            return None
        return worker_id

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
            entry.person_box = person.box
            identities.append(WorkerIdentity(
                worker_id=entry.worker_id,
                person=person,
                source="qr",
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
                source="qr",
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
    """Temporal warning for people without a visible/cached worker QR.

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
