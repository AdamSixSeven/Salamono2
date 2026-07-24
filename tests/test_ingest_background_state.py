from types import SimpleNamespace

import numpy as np
import pytest

from backend.frame_processor import FrameJob
from backend.models import FrameResultOut
from backend.routes import ingest


class _NoCalibration:
    @staticmethod
    def get(_camera_id):
        return None


class _RecordingManager:
    def __init__(self):
        self.frames = []

    async def broadcast_frame(self, data, jpeg_bytes=None):
        self.frames.append((data, jpeg_bytes))


def _app():
    return SimpleNamespace(state=SimpleNamespace(
        camera_registry={},
        calibration_store=_NoCalibration(),
        ws_manager=_RecordingManager(),
    ))


def _job(received_at: float, captured_at: float) -> FrameJob:
    return FrameJob(
        camera_id="cam",
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
        timestamp=captured_at,
        payload=ingest._FramePayload(
            mode="site",
            received_at=received_at,
            preview_jpeg_bytes=b"jpeg",
            preview_jpeg_b64="",
        ),
    )


def _result(frame_id: int, captured_at: float) -> FrameResultOut:
    return FrameResultOut(
        frame_id=frame_id,
        timestamp=captured_at,
        camera_id="cam",
        mode="site",
        detections=[],
        frame_jpeg_b64="",
        processing_ms=10.0,
    )


@pytest.mark.asyncio
async def test_current_background_failure_clears_pending_and_records_error():
    app = _app()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    ingest._record_frame_received(
        app,
        camera_id="cam",
        timestamp=10.0,
        received_at=20.0,
        mode="site",
        frame=frame,
        processing_pending=True,
    )

    await ingest.record_frame_failure(
        app,
        _job(received_at=20.0, captured_at=10.0),
        RuntimeError("model failed"),
    )

    row = app.state.camera_registry["cam"]
    assert row["processing_pending"] is False
    assert row["processing_error"] == "RuntimeError: model failed"
    assert row["last_failed_received_at"] == 20.0


@pytest.mark.asyncio
async def test_stale_completion_cannot_overwrite_or_publish_newer_result():
    app = _app()
    newer_job = _job(received_at=30.0, captured_at=300.0)
    older_job = _job(received_at=20.0, captured_at=200.0)

    await ingest.publish_frame_result(app, newer_job, _result(2, 300.0))
    await ingest.publish_frame_result(app, older_job, _result(1, 200.0))

    row = app.state.camera_registry["cam"]
    assert row["frames"] == 2
    assert row["last_processed_received_at"] == 30.0
    assert row["processed_captured_at"] == 300.0
    assert [data["frame_id"] for data, _jpeg in app.state.ws_manager.frames] == [2]

