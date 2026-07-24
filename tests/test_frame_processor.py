import asyncio
import threading
import time

import numpy as np
import pytest

from backend.frame_processor import (
    FrameJob,
    FrameSubmitStatus,
    LatestFrameProcessor,
)


def _job(camera_id: str, value: int) -> FrameJob:
    return FrameJob(
        camera_id=camera_id,
        frame=np.full((8, 8, 3), value, dtype=np.uint8),
        timestamp=float(value),
        payload=value,
    )


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_latest_pending_frame_wins_without_growing_queue():
    first_started = threading.Event()
    release_first = threading.Event()
    published: list[int] = []

    def process(job):
        if job.payload == 1:
            first_started.set()
            assert release_first.wait(timeout=2.0)
        return job.payload

    async def publish(_job, result):
        published.append(result)

    processor = LatestFrameProcessor(process, publish)
    await processor.start()
    try:
        first = await processor.submit(_job("cam", 1))
        await _wait_until(first_started.is_set)
        second = await processor.submit(_job("cam", 2))
        third = await processor.submit(_job("cam", 3))
        fourth = await processor.submit(_job("cam", 4))

        assert first.status is FrameSubmitStatus.ACCEPTED
        assert first.queue_depth == 1
        assert second.status is FrameSubmitStatus.ACCEPTED
        assert second.queue_depth == 1
        assert third.status is FrameSubmitStatus.REPLACED
        assert third.queue_depth == 1
        assert fourth.status is FrameSubmitStatus.REPLACED
        assert fourth.queue_depth == 1
        assert processor.queue_depth == 1
        assert processor.pending_count == 1

        release_first.set()
        await processor.drain()
        assert processor.queue_depth == 0
        assert published == [1, 4]
        assert processor.stats.submitted == 4
        assert processor.stats.replaced == 2
        assert processor.stats.processed == 2
        assert processor.stats.failed == 0
        assert processor.stats.dropped == 0
    finally:
        release_first.set()
        await processor.close()


@pytest.mark.asyncio
async def test_slow_camera_does_not_block_another_camera():
    camera_a_started = threading.Event()
    release_camera_a = threading.Event()
    camera_b_published = asyncio.Event()
    published: list[str] = []

    def process(job):
        if job.camera_id == "a":
            camera_a_started.set()
            assert release_camera_a.wait(timeout=2.0)
        return job.camera_id

    async def publish(_job, result):
        published.append(result)
        if result == "b":
            camera_b_published.set()

    processor = LatestFrameProcessor(process, publish)
    await processor.start()
    try:
        await processor.submit(_job("a", 1))
        await _wait_until(camera_a_started.is_set)
        await processor.submit(_job("b", 2))

        await asyncio.wait_for(camera_b_published.wait(), timeout=1.0)
        assert published == ["b"]

        release_camera_a.set()
        await processor.drain()
        assert published == ["b", "a"]
    finally:
        release_camera_a.set()
        await processor.close()


@pytest.mark.asyncio
async def test_serial_pipeline_keeps_waiting_camera_frame_replaceable():
    camera_a_started = threading.Event()
    release_camera_a = threading.Event()
    published: list[tuple[str, int]] = []

    def process(job):
        if job.camera_id == "a":
            camera_a_started.set()
            assert release_camera_a.wait(timeout=2.0)
        return job.camera_id, job.payload

    async def publish(_job, result):
        published.append(result)

    processor = LatestFrameProcessor(
        process,
        publish,
        max_parallel_cameras=1,
    )
    await processor.start()
    try:
        await processor.submit(_job("a", 1))
        await _wait_until(camera_a_started.is_set)
        await processor.submit(_job("b", 2))
        replacement = await processor.submit(_job("b", 3))

        assert replacement.status is FrameSubmitStatus.REPLACED
        assert processor.queue_depth == 1

        release_camera_a.set()
        await processor.drain()
        assert published == [("a", 1), ("b", 3)]
        assert processor.stats.replaced == 1
    finally:
        release_camera_a.set()
        await processor.close()


@pytest.mark.asyncio
async def test_submit_does_not_wait_for_heavy_callback():
    processing_started = threading.Event()
    release_processing = threading.Event()

    def process(job):
        processing_started.set()
        assert release_processing.wait(timeout=2.0)
        return job.payload

    async def publish(_job, _result):
        return None

    processor = LatestFrameProcessor(process, publish)
    await processor.start()
    try:
        before = time.perf_counter()
        result = await processor.submit(_job("cam", 1))
        elapsed = time.perf_counter() - before

        assert result.accepted
        assert elapsed < 0.1
        await _wait_until(processing_started.is_set)

        # The event loop remains responsive while process() waits in a thread.
        loop_tick = False

        async def tick():
            nonlocal loop_tick
            await asyncio.sleep(0)
            loop_tick = True

        await asyncio.wait_for(tick(), timeout=0.2)
        assert loop_tick
    finally:
        release_processing.set()
        await processor.close()


@pytest.mark.asyncio
async def test_processing_exception_is_counted_and_newest_job_still_runs():
    failing_started = threading.Event()
    release_failure = threading.Event()
    published: list[str] = []
    errors: list[tuple[str, str]] = []

    def process(job):
        if job.payload == "bad":
            failing_started.set()
            assert release_failure.wait(timeout=2.0)
            raise ValueError("corrupt frame")
        return str(job.payload)

    async def publish(_job, result):
        published.append(result)

    async def on_error(job, exc):
        errors.append((str(job.payload), str(exc)))

    processor = LatestFrameProcessor(process, publish, on_error=on_error)
    await processor.start()
    try:
        bad = FrameJob("cam", np.zeros((4, 4, 3), np.uint8), payload="bad")
        good = FrameJob("cam", np.ones((4, 4, 3), np.uint8), payload="good")
        await processor.submit(bad)
        await _wait_until(failing_started.is_set)
        await processor.submit(good)
        release_failure.set()

        await processor.drain()
        assert published == ["good"]
        assert errors == [("bad", "corrupt frame")]
        assert processor.stats.failed == 1
        assert processor.stats.processed == 1
    finally:
        release_failure.set()
        await processor.close()


@pytest.mark.asyncio
async def test_publish_exception_does_not_kill_camera_worker():
    failing_publish_started = asyncio.Event()
    release_failure = asyncio.Event()
    published: list[int] = []

    def process(job):
        return job.payload

    async def publish(_job, result):
        if result == 1:
            failing_publish_started.set()
            await release_failure.wait()
            raise RuntimeError("websocket disconnected")
        published.append(result)

    processor = LatestFrameProcessor(process, publish)
    await processor.start()
    try:
        await processor.submit(_job("cam", 1))
        await asyncio.wait_for(failing_publish_started.wait(), timeout=1.0)
        await processor.submit(_job("cam", 2))
        release_failure.set()

        await processor.drain()
        assert published == [2]
        assert processor.stats.failed == 1
        assert processor.stats.processed == 1
    finally:
        release_failure.set()
        await processor.close()


@pytest.mark.asyncio
async def test_submit_copies_frame_before_caller_can_mutate_it():
    processing_started = threading.Event()
    inspect_frame = threading.Event()
    observed_sums: list[int] = []

    def process(job):
        processing_started.set()
        assert inspect_frame.wait(timeout=2.0)
        return int(job.frame.sum())

    async def publish(_job, result):
        observed_sums.append(result)

    processor = LatestFrameProcessor(process, publish)
    await processor.start()
    original = np.zeros((32, 32, 3), dtype=np.uint8)
    try:
        await processor.submit(FrameJob("cam", original))
        await _wait_until(processing_started.is_set)
        original[:] = 255
        inspect_frame.set()

        await processor.drain()
        assert observed_sums == [0]
    finally:
        inspect_frame.set()
        await processor.close()


@pytest.mark.asyncio
async def test_busy_camera_capacity_drops_new_camera_then_idle_lru_is_evicted():
    camera_a_started = threading.Event()
    release_camera_a = threading.Event()
    published: list[str] = []

    def process(job):
        if job.camera_id == "a":
            camera_a_started.set()
            assert release_camera_a.wait(timeout=2.0)
        return job.camera_id

    async def publish(_job, result):
        published.append(result)

    processor = LatestFrameProcessor(process, publish, max_cameras=1)
    await processor.start()
    try:
        await processor.submit(_job("a", 1))
        await _wait_until(camera_a_started.is_set)
        rejected = await processor.submit(_job("b", 2))
        assert rejected.dropped
        assert rejected.reason == "camera_capacity_busy"
        assert processor.camera_count == 1

        release_camera_a.set()
        await processor.drain()

        # A is idle now, so the one-entry LRU may replace it with B.
        accepted = await processor.submit(_job("b", 3))
        assert accepted.accepted
        await processor.drain()
        assert published == ["a", "b"]
        assert processor.camera_count == 1
        assert processor.stats.dropped == 1
    finally:
        release_camera_a.set()
        await processor.close()


@pytest.mark.asyncio
async def test_close_without_drain_discards_pending_but_finishes_active_job():
    active_started = threading.Event()
    release_active = threading.Event()
    published: list[int] = []

    def process(job):
        if job.payload == 1:
            active_started.set()
            assert release_active.wait(timeout=2.0)
        return job.payload

    async def publish(_job, result):
        published.append(result)

    processor = LatestFrameProcessor(process, publish)
    await processor.start()
    await processor.submit(_job("cam", 1))
    await _wait_until(active_started.is_set)
    await processor.submit(_job("cam", 2))

    close_task = asyncio.create_task(processor.close(drain=False))
    await asyncio.sleep(0)
    assert not close_task.done()
    release_active.set()
    await asyncio.wait_for(close_task, timeout=1.0)

    assert published == [1]
    assert processor.stats.processed == 1
    assert processor.stats.dropped == 1

    rejected = await processor.submit(_job("cam", 3))
    assert rejected.dropped
    assert rejected.reason == "processor_closed"
