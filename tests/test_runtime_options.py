import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
