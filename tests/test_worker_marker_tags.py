import cv2
import numpy as np

from backend.detector import Detection
from backend.worker_identification import WorkerIdentifier, WorkerIDWorker
from backend.worker_tags import render_worker_marker, worker_marker_id
from config import WorkerIDConfig


def person(box=(100, 60, 300, 420)):
    return Detection(0, "person", "person", box, 0.95)


def test_simple_worker_marker_decodes_to_expected_worker_id(tmp_path):
    marker_id = worker_marker_id("W-002")
    marker = render_worker_marker(marker_id, 220)
    marker = cv2.copyMakeBorder(marker, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    y, x = 120, 160
    frame[y:y+marker.shape[0], x:x+marker.shape[1]] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    cfg = WorkerIDConfig(enabled=True, sample_fps=30.0, database_path=str(tmp_path / "workers.sqlite3"))
    # Pre-register worker in the same SQLite file so the marker resolves to the real ID.
    from backend.worker_store import WorkerStore
    WorkerStore(cfg.database_path).create("W-002", "Jan", "Kowalski", "Operator", "Budowa")

    identifier = WorkerIdentifier(cfg)
    identities = identifier.process("cam", frame, [person((120, 80, 420, 420))], 1.0)
    assert len(identities) == 1
    assert identities[0].worker_id == "W-002"
    assert identities[0].source == "marker"


def test_async_full_frame_worker_preserves_existing_aruco_dictionary_and_id(tmp_path):
    marker_id = worker_marker_id("W-030")
    marker = render_worker_marker(marker_id, 220)
    marker = cv2.copyMakeBorder(
        marker,
        40,
        40,
        40,
        40,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    frame[120:120 + marker.shape[0], 160:160 + marker.shape[1]] = (
        cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    )
    cfg = WorkerIDConfig(
        enabled=True,
        sample_fps=30.0,
        crop_padding=0.0,
        database_path=str(tmp_path / "workers.sqlite3"),
        initial_confirmations=1,
    )
    from backend.worker_store import WorkerStore
    WorkerStore(cfg.database_path).create(
        "W-030",
        "Anna",
        "Nowak",
        "Operator",
        "Budowa",
    )
    worker = WorkerIDWorker(WorkerIdentifier(cfg), cfg)
    try:
        people = worker.submit_latest(
            "cam-phone",
            frame,
            [person((120, 80, 500, 450))],
            1.0,
        )
        assert worker.wait_for_result("cam-phone", timeout=2.0) is not None
        identities = worker.current_identities(
            "cam-phone",
            people,
            1.1,
        )
        assert len(identities) == 1
        assert identities[0].worker_id == "W-030"
        assert identities[0].track_id == people[0].track_id == 1
        assert identities[0].tag_polygon == []
    finally:
        worker.close()
