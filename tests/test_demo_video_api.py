import cv2
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import numpy as np
import pytest

from backend.demo_video import DemoVideoService
from backend.routes.demo_videos import router
from config import DemoVideoConfig


class _Capture:
    def __init__(self, frame_count=3, fps=30.0):
        self.frames = [
            np.full((12, 16, 3), index, dtype=np.uint8)
            for index in range(frame_count)
        ]
        self.fps = fps
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position].copy()
        self.position += 1
        return True, frame

    def grab(self):
        if self.position >= len(self.frames):
            return False
        self.position += 1
        return True

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return 16
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return 12
        return 0

    def release(self):
        self.released = True


def _service(tmp_path, *, max_size_bytes=1024):
    cfg = DemoVideoConfig(
        upload_dir=str(tmp_path / "uploads"),
        output_dir=str(tmp_path / "outputs"),
        max_size_mb=1,
        job_ttl_seconds=60,
        max_pending_jobs=2,
        processing_enabled=True,
        playback_mode="fast",
        probe_timeout_seconds=0.5,
        control_timeout_seconds=0.5,
    )
    return DemoVideoService(
        cfg,
        capture_factory=lambda _path: _Capture(),
        max_size_bytes=max_size_bytes,
        start_janitor=False,
    )


def _app(service):
    app = FastAPI()
    app.state.demo_video_service = service
    app.include_router(router, prefix="/api")
    return app


@pytest.mark.asyncio
async def test_upload_status_play_and_delete_api(tmp_path):
    service = _service(tmp_path)
    app = _app(service)
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            upload = await client.post(
                "/api/demo-videos",
                files={"video": ("walk.mp4", b"whole-video", "video/mp4")},
                data={
                    "camera_id": "demo_upload",
                    "mode": "site",
                    "playback_mode": "fast",
                },
            )
            assert upload.status_code == 200
            created = upload.json()
            assert created["status"] == "uploaded"
            assert created["filename"] == "walk.mp4"
            assert created["size_bytes"] == len(b"whole-video")
            assert "input_path" not in created

            job_id = created["job_id"]
            service.wait_for_status(job_id, {"ready", "failed"}, timeout=1.0)
            status = await client.get(f"/api/demo-videos/{job_id}")
            assert status.status_code == 200
            assert status.json()["status"] == "ready"
            assert set(status.json()["metrics"]) >= {
                "demo_upload_ms",
                "demo_probe_ms",
                "demo_active_jobs",
            }

            playing = await client.post(f"/api/demo-videos/{job_id}/play")
            assert playing.status_code == 200
            assert playing.json()["status"] in {"playing", "finished"}
            service.wait_for_status(job_id, "finished", timeout=1.0)

            deleted = await client.delete(f"/api/demo-videos/{job_id}")
            assert deleted.status_code == 204
            assert (await client.get(f"/api/demo-videos/{job_id}")).status_code == 404
    finally:
        service.close(timeout=1.0)


@pytest.mark.asyncio
async def test_upload_api_rejects_type_and_size_without_leaking_jobs(tmp_path):
    service = _service(tmp_path, max_size_bytes=4)
    app = _app(service)
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            invalid = await client.post(
                "/api/demo-videos",
                files={"video": ("clip.txt", b"123", "text/plain")},
            )
            assert invalid.status_code == 415

            too_large = await client.post(
                "/api/demo-videos",
                files={"video": ("clip.mp4", b"12345", "video/mp4")},
            )
            assert too_large.status_code == 413

        assert service.list_jobs() == []
        assert list(service.upload_dir.iterdir()) == []
    finally:
        service.close(timeout=1.0)
