"""Bounded asynchronous latest-frame processing.

The HTTP ingest path can submit a frame quickly while expensive inference runs
in a worker thread.  Each camera owns at most one pending slot: when more
frames arrive during processing, only the newest one survives.

This module deliberately has no dependency on FastAPI or application state.
Lifecycle integration belongs in ``backend.main`` / ``backend.routes.ingest``.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from enum import Enum
import time
from typing import Any, Awaitable, Callable, Generic, TypeVar

import numpy as np


ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class FrameJob:
    """One immutable processing request.

    ``frame`` itself is a mutable NumPy array, therefore ``submit`` always
    creates a detached copy before exposing the job to a worker.  ``payload``
    may carry request-specific immutable metadata needed by an integration.
    """

    camera_id: str
    frame: np.ndarray
    timestamp: float = field(default_factory=time.time)
    payload: Any = None

    def detached(self) -> "FrameJob":
        return replace(self, frame=np.array(self.frame, copy=True, order="C"))


class FrameSubmitStatus(str, Enum):
    ACCEPTED = "accepted"
    REPLACED = "replaced"
    DROPPED = "dropped"


@dataclass(frozen=True, slots=True)
class FrameSubmitResult:
    camera_id: str
    status: FrameSubmitStatus
    reason: str | None = None
    queue_depth: int = 0

    @property
    def accepted(self) -> bool:
        return self.status is not FrameSubmitStatus.DROPPED

    @property
    def replaced(self) -> bool:
        return self.status is FrameSubmitStatus.REPLACED

    @property
    def dropped(self) -> bool:
        return self.status is FrameSubmitStatus.DROPPED


@dataclass(frozen=True, slots=True)
class FrameProcessorStats:
    submitted: int = 0
    replaced: int = 0
    dropped: int = 0
    processed: int = 0
    failed: int = 0


@dataclass
class _CameraSlot:
    pending: FrameJob | None = None
    worker: asyncio.Task[None] | None = None


class LatestFrameProcessor(Generic[ResultT]):
    """Process the newest available frame independently for each camera.

    ``process`` is a synchronous, potentially expensive callback and is always
    invoked through :func:`asyncio.to_thread`.  ``publish`` is awaited on the
    event loop after successful processing. ``max_parallel_cameras`` can
    serialize a shared GPU pipeline while leaving other cameras' pending
    frames replaceable; omitting it allows independent callbacks in parallel.

    Replaced frames are counted in ``replaced``; ``dropped`` is reserved for a
    rejected submission (capacity/closed processor) or a pending frame
    discarded by ``close(drain=False)``.
    """

    def __init__(
        self,
        process: Callable[[FrameJob], ResultT],
        publish: Callable[[FrameJob, ResultT], Awaitable[None]],
        *,
        max_cameras: int = 16,
        max_parallel_cameras: int | None = None,
        on_error: Callable[[FrameJob, BaseException], Awaitable[None]] | None = None,
    ):
        if max_cameras < 1:
            raise ValueError("max_cameras must be at least 1")
        if max_parallel_cameras is not None and max_parallel_cameras < 1:
            raise ValueError("max_parallel_cameras must be at least 1")
        if not callable(process) or not callable(publish):
            raise TypeError("process and publish must be callable")
        if on_error is not None and not callable(on_error):
            raise TypeError("on_error must be callable")

        self._process = process
        self._publish = publish
        self._on_error = on_error
        self._max_cameras = int(max_cameras)
        parallel_limit = (
            self._max_cameras
            if max_parallel_cameras is None
            else int(max_parallel_cameras)
        )
        self._processing_slots = asyncio.Semaphore(parallel_limit)
        self._slots: OrderedDict[str, _CameraSlot] = OrderedDict()
        self._lock = asyncio.Lock()
        self._changed = asyncio.Condition(self._lock)
        self._started = False
        self._closed = False
        self._submitted = 0
        self._replaced = 0
        self._dropped = 0
        self._processed = 0
        self._failed = 0
        self._pending_count = 0

    async def start(self) -> None:
        """Start accepting jobs.  Calling ``start`` twice is harmless."""
        async with self._lock:
            if self._closed:
                raise RuntimeError("frame processor has already been closed")
            self._started = True

    async def submit(self, job: FrameJob) -> FrameSubmitResult:
        """Store ``job`` as the latest pending frame without running inference.

        The frame is copied before the method returns.  Apart from that bounded
        memory copy, this method performs no CPU-heavy work.
        """
        if not isinstance(job, FrameJob):
            raise TypeError("job must be a FrameJob")
        if not str(job.camera_id).strip():
            raise ValueError("camera_id must not be empty")
        if not isinstance(job.frame, np.ndarray) or job.frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array")

        # Detach outside the scheduler lock and outside the event-loop thread.
        # A 960×720 BGR copy is bounded but still large enough to create a
        # visible scheduling hitch when several cameras submit together.
        queued = await asyncio.to_thread(job.detached)
        camera_id = str(queued.camera_id)

        async with self._lock:
            if self._closed:
                self._submitted += 1
                self._dropped += 1
                return FrameSubmitResult(
                    camera_id,
                    FrameSubmitStatus.DROPPED,
                    "processor_closed",
                    self._pending_count,
                )
            if not self._started:
                raise RuntimeError("frame processor has not been started")

            self._submitted += 1

            slot = self._slots.get(camera_id)
            if slot is None:
                slot = self._make_room_locked(camera_id)
                if slot is None:
                    self._dropped += 1
                    return FrameSubmitResult(
                        camera_id,
                        FrameSubmitStatus.DROPPED,
                        "camera_capacity_busy",
                        self._pending_count,
                    )
            else:
                self._slots.move_to_end(camera_id)

            status = FrameSubmitStatus.ACCEPTED
            if slot.pending is not None:
                self._replaced += 1
                status = FrameSubmitStatus.REPLACED
            else:
                self._pending_count += 1
            slot.pending = queued

            if slot.worker is None or slot.worker.done():
                slot.worker = asyncio.create_task(
                    self._camera_worker(camera_id, slot),
                    name=f"latest-frame-{camera_id}",
                )
            self._changed.notify_all()
            return FrameSubmitResult(
                camera_id,
                status,
                queue_depth=self._pending_count,
            )

    def _make_room_locked(self, camera_id: str) -> _CameraSlot | None:
        if len(self._slots) >= self._max_cameras:
            # An idle entry has no data to lose and can be evicted using LRU
            # order.  Running cameras are never cancelled because to_thread
            # work cannot be stopped safely.
            evict_id = next(
                (
                    existing_id
                    for existing_id, existing in self._slots.items()
                    if existing.pending is None
                    and (existing.worker is None or existing.worker.done())
                ),
                None,
            )
            if evict_id is None:
                return None
            self._slots.pop(evict_id, None)

        slot = _CameraSlot()
        self._slots[camera_id] = slot
        return slot

    async def _camera_worker(
        self,
        camera_id: str,
        slot: _CameraSlot,
    ) -> None:
        while True:
            # Waiting cameras deliberately leave their frame in the replaceable
            # pending slot.  With max_parallel_cameras=1, a frame therefore
            # cannot become stale merely by waiting behind another camera's
            # GPU inference.
            async with self._processing_slots:
                async with self._lock:
                    # The slot cannot be evicted while this task is alive.
                    if slot.pending is None:
                        slot.worker = None
                        self._slots.move_to_end(camera_id)
                        self._changed.notify_all()
                        return
                    job = slot.pending
                    slot.pending = None
                    self._pending_count -= 1
                    self._slots.move_to_end(camera_id)
                    self._changed.notify_all()

                try:
                    result = await asyncio.to_thread(self._process, job)
                    await self._publish(job, result)
                except Exception as exc:
                    # One corrupt frame or a transient publication failure must
                    # not kill this camera's worker or strand its newest frame.
                    async with self._lock:
                        self._failed += 1
                        self._changed.notify_all()
                    if self._on_error is not None:
                        try:
                            await self._on_error(job, exc)
                        except Exception:
                            # Error reporting is best-effort; the worker must
                            # still continue to its newest pending frame.
                            pass
                else:
                    async with self._lock:
                        self._processed += 1
                        self._changed.notify_all()

    async def drain(self) -> None:
        """Wait until all accepted/replacement jobs have reached a terminal state."""
        async with self._changed:
            await self._changed.wait_for(self._is_idle_locked)

    def _is_idle_locked(self) -> bool:
        return all(
            slot.pending is None
            and (slot.worker is None or slot.worker.done())
            for slot in self._slots.values()
        )

    async def close(self, *, drain: bool = True) -> None:
        """Stop accepting work and finish active workers.

        With ``drain=False`` pending slots are discarded, but an already
        running ``to_thread`` callback is still allowed to finish safely.
        """
        async with self._lock:
            if self._closed:
                # A prior close may still be draining an active callback.
                pass
            else:
                self._closed = True
                if not drain:
                    for slot in self._slots.values():
                        if slot.pending is not None:
                            slot.pending = None
                            self._pending_count -= 1
                            self._dropped += 1
                self._changed.notify_all()

        await self.drain()
        async with self._lock:
            self._slots.clear()
            self._started = False
            self._changed.notify_all()

    @property
    def stats(self) -> FrameProcessorStats:
        # Counters are only mutated on the event-loop thread.  Returning a
        # frozen snapshot prevents consumers from changing internal state.
        return FrameProcessorStats(
            submitted=self._submitted,
            replaced=self._replaced,
            dropped=self._dropped,
            processed=self._processed,
            failed=self._failed,
        )

    @property
    def camera_count(self) -> int:
        return len(self._slots)

    @property
    def queue_depth(self) -> int:
        """Current pending-frame count; actively processed jobs are excluded.

        The value is maintained under ``_lock`` rather than calculated by
        iterating mutable slots, making this an inexpensive safe snapshot for
        status/readiness responses.
        """
        return self._pending_count

    @property
    def pending_count(self) -> int:
        """Backward-friendly alias for :attr:`queue_depth`."""
        return self._pending_count

    async def __aenter__(self) -> "LatestFrameProcessor[ResultT]":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()
