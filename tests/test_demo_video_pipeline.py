from types import SimpleNamespace

import numpy as np
import pytest

from backend.calibration import CalibrationStore
from backend.camera_registry import CameraRegistry
from backend.latest_frame_store import LatestFrameStore
from backend.models import FrameResultOut
from backend.preview_encoder import PreviewJpegEncoder
from backend.routes.ingest import create_decoded_frame_job, publish_frame_result


class _WsManager:
    def __init__(self):
        self.frames = []

    async def broadcast_frame(self, data, jpeg_bytes):
        self.frames.append((data, jpeg_bytes))


def _app(tmp_path):
    state = SimpleNamespace(
        ppe_detector=None,
        latest_frame_store=LatestFrameStore(max_cameras=2),
        camera_registry=CameraRegistry(max_cameras=2),
        calibration_store=CalibrationStore(str(tmp_path / "calibration.json")),
        ws_manager=_WsManager(),
    )
    return SimpleNamespace(state=state)


@pytest.mark.asyncio
async def test_decoded_demo_frame_uses_matching_binary_pipeline_metadata(tmp_path):
    app = _app(tmp_path)
    frame = np.full((24, 32, 3), 80, dtype=np.uint8)
    job = create_decoded_frame_job(
        app,
        frame=frame,
        camera_id="demo_upload",
        timestamp=1_800_000_000.25,
        mode="site",
    )
    result = FrameResultOut(
        frame_id=9,
        timestamp=job.timestamp,
        camera_id=job.camera_id,
        detections=[],
        frame_jpeg_b64="",
        processing_ms=3.0,
    )

    await publish_frame_result(
        app,
        job,
        result,
        extra_metadata={
            "type": "frame",
            "job_id": "job-a",
            "run_id": "run-b",
            "frame_index": 4,
            "source_time_sec": 0.16,
            "status": "playing",
            "processing_ms": 4.5,
            "processing_fps": 22.2,
        },
    )

    assert len(app.state.ws_manager.frames) == 1
    metadata, jpeg = app.state.ws_manager.frames[0]
    assert jpeg.startswith(b"\xff\xd8")
    assert metadata["camera_id"] == "demo_upload"
    assert metadata["job_id"] == "job-a"
    assert metadata["run_id"] == "run-b"
    assert metadata["frame_index"] == 4
    assert metadata["source_time_sec"] == 0.16
    assert metadata["processing_ms"] == 4.5

    latest = app.state.latest_frame_store.get("demo_upload")
    assert latest is not None
    assert latest.timestamp == job.timestamp
    assert latest.image_bytes == jpeg
    assert app.state.camera_registry["demo_upload"]["frames_received"] == 1
    assert app.state.camera_registry["demo_upload"]["processing_pending"] is False


@pytest.mark.asyncio
async def test_demo_jpeg_can_overlap_analysis_and_still_matches_metadata(tmp_path):
    app = _app(tmp_path)
    app.state.preview_encoder = PreviewJpegEncoder()
    try:
        frame = np.full((240, 320, 3), 120, dtype=np.uint8)
        job = create_decoded_frame_job(
            app,
            frame=frame,
            camera_id="demo_async",
            timestamp=1_800_000_001.0,
        )
        assert job.payload.preview_jpeg_bytes == b""
        assert job.payload.preview_jpeg_future is not None

        result = FrameResultOut(
            frame_id=10,
            timestamp=job.timestamp,
            camera_id=job.camera_id,
            detections=[],
            frame_jpeg_b64="",
            processing_ms=2.0,
        )
        await publish_frame_result(app, job, result)
        metadata, jpeg = app.state.ws_manager.frames[0]
        assert metadata["camera_id"] == "demo_async"
        assert jpeg.startswith(b"\xff\xd8")
    finally:
        app.state.preview_encoder.close()
