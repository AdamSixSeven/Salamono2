import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time
from unittest.mock import MagicMock

import cv2
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.alert_storage import AlertStore
from backend.calibration import CalibrationStore
from backend.danger_rules import DangerDetector, TemporalFilter
from backend.detector import Detection
from backend.frame_store import FrameStore
from backend.marker_detector import MarkerDetector
from backend.models import AlarmRecord, AlertSeverity
from backend.routes import ingest, workers
from backend.worker_identification import WorkerIdentity
from backend.worker_store import DuplicateWorkerError, WorkerStore
from backend.ws_manager import ConnectionManager
from backend.zone_rules import ZoneBreachDetector, ZoneTemporalFilter
from backend.zones_store import ZoneStore


def _request(app: FastAPI, method: str, path: str, **kwargs):
    async def run():
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


def _profile(worker_id: str = "W-001") -> dict[str, str]:
    return {
        "worker_id": worker_id,
        "first_name": "Jan",
        "last_name": "Kowalski",
        "position": "Operator koparki",
        "department": "Roboty ziemne",
    }


def test_worker_store_is_empty_and_persists_crud(tmp_path):
    path = tmp_path / "workers.sqlite3"
    store = WorkerStore(str(path))
    assert store.list() == []

    created = store.create(**_profile())
    assert created.worker_id == "W-001"
    assert created.full_name == "Jan Kowalski"
    assert path.exists()

    # A new repository instance reads the same on-disk database.
    reopened = WorkerStore(str(path))
    assert reopened.get("W-001") == created

    try:
        reopened.create(**_profile())
        raise AssertionError("duplicate worker ID should fail")
    except DuplicateWorkerError:
        pass

    updated = reopened.update(
        "W-001",
        first_name="Jan",
        last_name="Kowalski",
        position="Brygadzista",
        department="Roboty ziemne",
    )
    assert updated is not None
    assert updated.position == "Brygadzista"
    assert updated.created_at == created.created_at
    assert updated.updated_at >= created.updated_at
    assert reopened.delete("W-001") is True
    assert reopened.delete("W-001") is False
    assert WorkerStore(str(path)).list() == []


def test_worker_store_supports_fastapi_threadpool_access(tmp_path):
    store = WorkerStore(str(tmp_path / "workers.sqlite3"))

    def create(index: int):
        return store.create(
            worker_id=f"W-{index:03d}",
            first_name=f"Imię {index}",
            last_name="Testowy",
            position="Operator",
            department="Testy",
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        created = list(executor.map(create, range(18)))
        loaded = list(executor.map(
            store.get,
            (worker.worker_id for worker in created),
        ))

    assert len(store.list()) == 18
    assert {worker.worker_id for worker in loaded if worker is not None} == {
        f"W-{index:03d}" for index in range(18)
    }


def test_workers_crud_api_and_static_summary_route(tmp_path):
    api = FastAPI()
    api.include_router(workers.router, prefix="/api")
    api.state.worker_store = WorkerStore(str(tmp_path / "workers.sqlite3"))
    api.state.alert_store = AlertStore(str(tmp_path / "alerts.jsonl"))

    payload = _profile()
    payload["worker_id"] = "  W-001  "
    response = _request(api, "POST", "/api/workers", json=payload)
    assert response.status_code == 201
    created = response.json()
    assert created["worker_id"] == "W-001"
    assert created["full_name"] == "Jan Kowalski"

    duplicate = _request(api, "POST", "/api/workers", json=_profile())
    assert duplicate.status_code == 409

    listed = _request(api, "GET", "/api/workers")
    assert [worker["worker_id"] for worker in listed.json()] == ["W-001"]

    fetched = _request(api, "GET", "/api/workers/W-001")
    assert fetched.status_code == 200
    assert fetched.json()["department"] == "Roboty ziemne"

    update_payload = {
        "first_name": "Jan",
        "last_name": "Kowalski",
        "position": "Brygadzista",
        "department": "Utrzymanie ruchu",
    }
    updated = _request(api, "PUT", "/api/workers/W-001", json=update_payload)
    assert updated.status_code == 200
    assert updated.json()["position"] == "Brygadzista"

    # This must resolve to the existing static endpoint, not worker_id=summary.
    summary = _request(api, "GET", "/api/workers/summary")
    assert summary.status_code == 200
    assert summary.json() == {"workers": []}

    deleted = _request(api, "DELETE", "/api/workers/W-001")
    assert deleted.status_code == 204
    assert _request(api, "GET", "/api/workers/W-001").status_code == 404


def test_worker_summary_is_enriched_but_unknown_ids_still_work(tmp_path):
    api = FastAPI()
    api.include_router(workers.router, prefix="/api")
    store = WorkerStore(str(tmp_path / "workers.sqlite3"))
    store.create(**_profile())
    api.state.worker_store = store
    api.state.alert_store = AlertStore(str(tmp_path / "alerts.jsonl"))

    known = AlarmRecord(
        id="known",
        timestamp=1000.0,
        mode="site",
        kind="site_hazard",
        severity=AlertSeverity.DANGER,
        rule_name="person_near_vehicle",
        description="test",
        details={"worker_id": "W-001"},
    )
    unknown = known.model_copy(deep=True)
    unknown.id = "unknown"
    unknown.details["worker_id"] = "W-999"
    api.state.alert_store.extend([known, unknown])

    rows = _request(api, "GET", "/api/workers/summary").json()["workers"]
    by_id = {row["worker_id"]: row for row in rows}
    assert by_id["W-001"]["full_name"] == "Jan Kowalski"
    assert by_id["W-001"]["department"] == "Roboty ziemne"
    assert "full_name" not in by_id["W-999"]

    # Marker generation remains valid even before a profile is created.
    tag = _request(api, "GET", "/api/worker-tag.png?worker_id=W-999")
    assert tag.status_code == 200
    assert tag.headers["content-type"].startswith("image/png")


def _jpeg() -> bytes:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", frame)
    return encoded.tobytes()


def test_frame_stream_and_alert_snapshot_include_registered_profile(tmp_path):
    api = FastAPI()
    api.include_router(ingest.router, prefix="/api")

    person = Detection(
        class_id=0,
        class_name="person",
        category="person",
        box=(150, 150, 250, 350),
        confidence=0.90,
        track_id=17,
    )
    vehicle = Detection(
        class_id=7,
        class_name="truck",
        category="vehicle",
        box=(100, 100, 300, 380),
        confidence=0.92,
    )
    detector = MagicMock()
    detector.detect.return_value = [person, vehicle]

    class FakeWorkerIdentifier:
        available = True

        @staticmethod
        def process(camera_id, frame, persons, timestamp):
            raise AssertionError("async Worker ID path must be used")

    class FakeWorkerIDWorker:
        class Stats:
            thread_alive = True

        @staticmethod
        def submit_latest(camera_id, frame, persons, timestamp):
            return persons

        @staticmethod
        def current_identities(camera_id, persons, timestamp):
            return [
                WorkerIdentity(
                    worker_id="W-001",
                    person=persons[0],
                    source="cache",
                    tag_polygon=[],
                    frame_timestamp=timestamp,
                    cached=True,
                    track_id=persons[0].track_id,
                )
            ]

        @staticmethod
        def scanned_track_ids(camera_id):
            return {17}

        @staticmethod
        def stats():
            return FakeWorkerIDWorker.Stats()

    class DisabledPosture:
        available = False

    store = WorkerStore(str(tmp_path / "workers.sqlite3"))
    store.create(**_profile())
    api.state.worker_store = store
    api.state.worker_identifier = FakeWorkerIdentifier()
    api.state.worker_id_worker = FakeWorkerIDWorker()
    api.state.unidentified_worker_monitor = None
    api.state.detector = detector
    api.state.danger_detector = DangerDetector()
    api.state.temporal_filter = TemporalFilter(required=1, cooldown_sec=0.0)
    api.state.ws_manager = ConnectionManager()
    api.state.frame_store = FrameStore(str(tmp_path / "flagged"))
    api.state.alert_store = AlertStore(str(tmp_path / "alerts.jsonl"))
    api.state.zone_store = ZoneStore(str(tmp_path / "zones.json"))
    api.state.zone_detector = ZoneBreachDetector()
    api.state.zone_temporal_filter = ZoneTemporalFilter(required=1, cooldown_sec=0.0)
    api.state.marker_detector = MarkerDetector()
    api.state.calibration_store = CalibrationStore(str(tmp_path / "calibration.json"))
    api.state.marker_zone_cache = {}
    api.state.debug_inject_person = None
    api.state.posture_manager = DisabledPosture()
    api.state.evidence_recorder = None
    api.state.frame_counter = 0
    api.state.camera_registry = {}
    api.state.start_time = time.time()

    response = _request(
        api,
        "POST",
        "/api/frame",
        files={"image": ("frame.jpg", _jpeg(), "image/jpeg")},
        data={"camera_id": "test", "timestamp": "1000.0"},
    )
    assert response.status_code == 200
    identification = response.json()["worker_identifications"][0]
    assert identification["worker_id"] == "W-001"
    assert identification["track_id"] == 17
    assert identification["person_box"] == [150, 150, 250, 350]
    assert identification["cached"] is True
    assert identification["registered"] is True
    assert identification["full_name"] == "Jan Kowalski"
    assert identification["position"] == "Operator koparki"
    assert identification["department"] == "Roboty ziemne"

    records = api.state.alert_store.query(worker_id="W-001")
    assert records
    assert records[0].details["worker_full_name"] == "Jan Kowalski"
    assert records[0].details["worker_department"] == "Roboty ziemne"
