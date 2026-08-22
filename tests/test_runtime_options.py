import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.routes.runtime import (
    RuntimeOptionsPatch,
    patch_runtime_options,
    reset_runtime_options,
)
from backend.runtime_options import RuntimeProcessingOptions, RuntimeProcessingStore


ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
APP = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")


def test_detector_dependency_graph_is_dynamic():
    options = RuntimeProcessingOptions(
        boxes=False,
        posture=False,
        zones=False,
        markers=True,
        distances=False,
        worker_id=False,
        ppe=False,
    )
    assert options.requires_detector("site") is False
    assert options.requires_detector("checkpoint") is False

    assert RuntimeProcessingOptions(boxes=True, posture=False, zones=False,
                                    markers=False, distances=False,
                                    worker_id=False, ppe=False).requires_detector("site")
    assert RuntimeProcessingOptions(boxes=False, posture=True, zones=False,
                                    markers=False, distances=False,
                                    worker_id=False, ppe=False).requires_detector("site")
    assert RuntimeProcessingOptions(boxes=False, posture=False, zones=False,
                                    markers=False, distances=False,
                                    worker_id=False, ppe=True).requires_detector("checkpoint")


def test_runtime_store_is_per_camera_and_resettable():
    store = RuntimeProcessingStore(max_cameras=4)
    first = store.update("cam-a", posture=False, worker_id=False)
    second = store.get("cam-b")

    assert first.posture is False
    assert first.worker_id is False
    assert second.posture is True
    assert second.worker_id is True

    reset = store.reset("cam-a")
    assert reset.posture is True
    assert reset.worker_id is True


def test_main_panel_sends_layer_switches_to_backend():
    assert 'data-layer="worker_id"' in INDEX
    assert '"/api/runtime/"' in APP
    assert "syncRuntimeProcessing" in APP
    assert "worker_id: !!State.layers.worker_id" in APP


class _RecordingWorkerIDWorker:
    def __init__(self):
        self.calls = []

    def set_camera_enabled(self, camera_id, enabled):
        self.calls.append((camera_id, enabled))


def _runtime_request(*, worker_id_worker=...):
    state = SimpleNamespace(
        runtime_processing_store=RuntimeProcessingStore(max_cameras=4),
    )
    if worker_id_worker is not ...:
        state.worker_id_worker = worker_id_worker
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_runtime_patch_disables_and_enables_worker_id_worker():
    worker = _RecordingWorkerIDWorker()
    request = _runtime_request(worker_id_worker=worker)

    disabled = asyncio.run(patch_runtime_options(
        "cam-worker",
        RuntimeOptionsPatch(worker_id=False),
        request,
    ))
    enabled = asyncio.run(patch_runtime_options(
        "cam-worker",
        RuntimeOptionsPatch(worker_id=True),
        request,
    ))

    assert disabled["options"]["worker_id"] is False
    assert enabled["options"]["worker_id"] is True
    assert worker.calls == [
        ("cam-worker", False),
        ("cam-worker", True),
    ]


def test_runtime_reset_restores_worker_id_worker_default():
    worker = _RecordingWorkerIDWorker()
    request = _runtime_request(worker_id_worker=worker)
    request.app.state.runtime_processing_store.update(
        "cam-worker",
        worker_id=False,
    )

    payload = asyncio.run(reset_runtime_options("cam-worker", request))

    assert payload["options"]["worker_id"] is True
    assert worker.calls == [("cam-worker", True)]


def test_runtime_worker_id_updates_are_safe_without_worker():
    request = _runtime_request()

    disabled = asyncio.run(patch_runtime_options(
        "cam-embedded",
        RuntimeOptionsPatch(worker_id=False),
        request,
    ))
    reset = asyncio.run(reset_runtime_options("cam-embedded", request))

    assert disabled["options"]["worker_id"] is False
    assert reset["options"]["worker_id"] is True
