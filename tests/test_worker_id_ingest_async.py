import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from backend.detector import Detection
from backend import main as main_module
from backend.routes.ingest import _identify_workers
from backend.worker_identification import (
    DecodedWorkerTag,
    WorkerIdentifier,
    WorkerIDWorker,
)
from backend.worker_store import WorkerStore
from config import WorkerIDConfig


WORKER_ID_METRIC_KEYS = {
    "worker_id_scan_ms",
    "worker_id_queue_age_ms",
    "worker_id_crop_count",
    "worker_id_cache_hits",
    "worker_id_cache_misses",
    "worker_id_dropped_jobs",
}


def _person(box=(20, 20, 80, 100)) -> Detection:
    return Detection(
        class_id=0,
        class_name="person",
        category="person",
        box=box,
        confidence=0.9,
    )


class _SlowDecoder:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def decode(self, frame):
        self.started.set()
        self.release.wait(0.75)
        return [
            DecodedWorkerTag(
                payload="marker:1",
                polygon=[(36, 36), (44, 36), (44, 44), (36, 44)],
            ),
        ]


def test_identify_workers_is_nonblocking_and_defers_monitor_until_first_scan(
    tmp_path,
):
    decoder = _SlowDecoder()
    config = WorkerIDConfig(
        enabled=True,
        sample_fps=30.0,
        cache_ttl_seconds=4.0,
        crop_padding=0.0,
        database_path=str(tmp_path / "workers.sqlite3"),
        initial_confirmations=1,
    )
    identifier = WorkerIdentifier(config, decoder=decoder)
    WorkerStore(config.database_path).create("W-001", "Jan", "Test", "", "")
    worker = WorkerIDWorker(
        identifier,
        config,
        # Scan once in this test. The second frame consumes the completed
        # identity cache without scheduling another decoder call.
        sample_fps=0.0,
        crop_padding=0.0,
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                worker_id_worker=worker,
                worker_identifier=identifier,
            ),
        ),
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    try:
        started_at = time.perf_counter()
        tracked, identities, monitor_persons = _identify_workers(
            request,
            "cam-phone",
            frame,
            [_person()],
            10.0,
        )
        elapsed = time.perf_counter() - started_at

        assert elapsed < 0.25
        assert decoder.started.wait(0.25)
        assert [person.track_id for person in tracked] == [1]
        assert identities == []
        assert monitor_persons == []

        decoder.release.set()
        assert worker.wait_for_result("cam-phone", timeout=1.0) is not None

        current_box = (24, 22, 84, 102)
        tracked, identities, monitor_persons = _identify_workers(
            request,
            "cam-phone",
            frame,
            [_person(current_box)],
            10.1,
        )

        assert [person.track_id for person in monitor_persons] == [1]
        assert len(identities) == 1
        assert identities[0].track_id == 1
        assert identities[0].person is tracked[0]
        assert identities[0].person.box == current_box
        assert identities[0].tag_polygon == []
        assert identities[0].cached is True
    finally:
        decoder.release.set()
        worker.close()


@pytest.mark.asyncio
async def test_readiness_exposes_exact_worker_id_metrics_and_profile_cache():
    current_app = main_module.app
    worker_stats = SimpleNamespace(
        scan_ms=12.3456,
        queue_age_ms=4.5678,
        crop_count=3,
        cache_hits=8,
        cache_misses=5,
        dropped_jobs=2,
        thread_alive=True,
    )
    profile_stats = SimpleNamespace(
        hits=13,
        misses=4,
        database_errors=1,
        stale_hits=2,
        invalidations=6,
        entries=9,
    )
    fake_worker = SimpleNamespace(stats=lambda: worker_stats)
    fake_profile_cache = SimpleNamespace(stats=lambda: profile_stats)

    sentinel = object()
    previous = {
        name: getattr(current_app.state, name, sentinel)
        for name in (
            "worker_identifier",
            "worker_id_worker",
            "worker_profile_cache",
        )
    }
    current_app.state.worker_identifier = SimpleNamespace(available=True)
    current_app.state.worker_id_worker = fake_worker
    current_app.state.worker_profile_cache = fake_profile_cache

    try:
        async with AsyncClient(
            transport=ASGITransport(app=current_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/readiness")

        assert response.status_code == 200
        data = response.json()
        expected_metrics = {
            "worker_id_scan_ms": 12.346,
            "worker_id_queue_age_ms": 4.568,
            "worker_id_crop_count": 3,
            "worker_id_cache_hits": 8,
            "worker_id_cache_misses": 5,
            "worker_id_dropped_jobs": 2,
        }
        assert set(data["worker_id_metrics"]) == WORKER_ID_METRIC_KEYS
        assert data["worker_id_metrics"] == expected_metrics
        assert {
            key: data[key] for key in WORKER_ID_METRIC_KEYS
        } == expected_metrics
        assert data["worker_id_profile_cache"] == {
            "hits": 13,
            "misses": 4,
            "database_errors": 1,
            "stale_hits": 2,
            "invalidations": 6,
            "entries": 9,
        }
    finally:
        for name, value in previous.items():
            if value is sentinel:
                delattr(current_app.state, name)
            else:
                setattr(current_app.state, name, value)
