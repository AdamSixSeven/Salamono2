import asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import time

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.alert_storage import AlertStore
from backend.routes import workers
from backend.worker_profile_cache import WorkerProfileCache
from backend.worker_store import WorkerRecord, WorkerStore


def _record(worker_id: str, position: str = "Operator") -> WorkerRecord:
    return WorkerRecord(
        worker_id=worker_id,
        first_name="Jan",
        last_name="Kowalski",
        position=position,
        department="Produkcja",
        created_at=1.0,
        updated_at=1.0,
    )


class _CountingStore:
    def __init__(self, records=()):
        self.records = {record.worker_id: record for record in records}
        self.get_calls: list[str] = []
        self.get_many_calls: list[list[str]] = []
        self.fail = False
        self.delay = 0.0

    def get(self, worker_id: str):
        self.get_calls.append(worker_id)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise sqlite3.OperationalError("database is temporarily locked")
        return self.records.get(worker_id)

    def get_many(self, worker_ids):
        worker_ids = list(worker_ids)
        self.get_many_calls.append(worker_ids)
        if self.fail:
            raise sqlite3.OperationalError("database is temporarily locked")
        return {
            worker_id: self.records[worker_id]
            for worker_id in worker_ids
            if worker_id in self.records
        }


def test_profile_cache_caches_positive_and_negative_results_until_ttl():
    now = [100.0]
    store = _CountingStore([_record("W-001")])
    cache = WorkerProfileCache(store, ttl_seconds=10.0, clock=lambda: now[0])

    assert cache.get("W-001") == store.records["W-001"]
    assert cache.get("W-001") == store.records["W-001"]
    assert cache.get("W-404") is None
    store.records["W-404"] = _record("W-404")
    assert cache.get("W-404") is None
    assert store.get_calls == ["W-001", "W-404"]

    now[0] = 110.0
    assert cache.get("W-001") == store.records["W-001"]
    assert cache.get("W-404") == store.records["W-404"]
    assert store.get_calls == ["W-001", "W-404", "W-001", "W-404"]
    assert cache.stats().hits == 2
    assert cache.stats().misses == 4


def test_profile_cache_get_many_batches_only_uncached_identifiers():
    now = [10.0]
    first = _record("W-001")
    second = _record("W-002")
    store = _CountingStore([first, second])
    cache = WorkerProfileCache(store, ttl_seconds=60.0, clock=lambda: now[0])

    assert cache.get("W-001") == first
    result = cache.get_many(["W-001", "W-002", "W-002", "W-404"])

    assert result == {"W-001": first, "W-002": second}
    assert store.get_calls == ["W-001"]
    assert store.get_many_calls == [["W-002", "W-404"]]

    # Both a found profile and a confirmed missing ID are now cache hits.
    assert cache.get_many(["W-002", "W-404"]) == {"W-002": second}
    assert len(store.get_many_calls) == 1
    stats = cache.stats()
    assert stats.hits == 3
    assert stats.misses == 3
    assert stats.entries == 3


def test_profile_cache_serves_stale_profile_during_temporary_sqlite_error():
    now = [0.0]
    old = _record("W-001", "Operator")
    store = _CountingStore([old])
    cache = WorkerProfileCache(store, ttl_seconds=10.0, clock=lambda: now[0])
    assert cache.get("W-001") == old

    now[0] = 11.0
    store.fail = True
    assert cache.get("W-001") == old
    assert cache.get("W-404") is None
    stats = cache.stats()
    assert stats.database_errors == 2
    assert stats.stale_hits == 1

    # Failed lookups have a short backoff, avoiding a SQLite/logging storm.
    assert cache.get("W-001") == old
    assert len(store.get_calls) == 3

    now[0] = 13.1
    updated = _record("W-001", "Brygadzista")
    store.records["W-001"] = updated
    store.fail = False
    assert cache.get("W-001") == updated


def test_profile_cache_invalidate_and_clear_force_reload():
    store = _CountingStore([_record("W-001"), _record("W-002")])
    cache = WorkerProfileCache(store, ttl_seconds=60.0)
    cache.get_many(["W-001", "W-002"])

    changed = _record("W-001", "Brygadzista")
    store.records["W-001"] = changed
    cache.invalidate("W-001")
    assert cache.get("W-001") == changed
    assert cache.get("W-002") == store.records["W-002"]

    cache.clear()
    cache.get_many(["W-001", "W-002"])
    assert len(store.get_many_calls) == 2
    stats = cache.stats()
    assert stats.invalidations == 2
    assert stats.entries == 2


def test_profile_cache_serializes_concurrent_first_lookup():
    store = _CountingStore([_record("W-001")])
    store.delay = 0.02
    cache = WorkerProfileCache(store, ttl_seconds=60.0)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(cache.get, ["W-001"] * 16))

    assert results == [store.records["W-001"]] * 16
    assert store.get_calls == ["W-001"]
    assert cache.stats().hits == 15
    assert cache.stats().misses == 1


def _request(app: FastAPI, method: str, path: str, **kwargs):
    async def run():
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


def test_worker_crud_routes_invalidate_profile_cache(tmp_path):
    class RecordingIdentifier:
        def __init__(self):
            self.invalidations = 0

        def invalidate_marker_index(self):
            self.invalidations += 1

    app = FastAPI()
    app.include_router(workers.router, prefix="/api")
    store = WorkerStore(str(tmp_path / "workers.sqlite3"))
    cache = WorkerProfileCache(store, ttl_seconds=600.0)
    app.state.worker_store = store
    app.state.worker_profile_cache = cache
    app.state.worker_identifier = RecordingIdentifier()
    app.state.alert_store = AlertStore(str(tmp_path / "alerts.jsonl"))

    # Prime a negative entry before the profile exists.
    assert cache.get("W-001") is None
    created = _request(
        app,
        "POST",
        "/api/workers",
        json={
            "worker_id": "W-001",
            "first_name": "Jan",
            "last_name": "Kowalski",
            "position": "Operator",
            "department": "Produkcja",
        },
    )
    assert created.status_code == 201
    assert cache.get("W-001").position == "Operator"

    updated = _request(
        app,
        "PUT",
        "/api/workers/W-001",
        json={
            "first_name": "Jan",
            "last_name": "Kowalski",
            "position": "Brygadzista",
            "department": "Produkcja",
        },
    )
    assert updated.status_code == 200
    assert cache.get("W-001").position == "Brygadzista"

    deleted = _request(app, "DELETE", "/api/workers/W-001")
    assert deleted.status_code == 204
    assert cache.get("W-001") is None
    assert cache.stats().invalidations == 3
    assert app.state.worker_identifier.invalidations == 3
