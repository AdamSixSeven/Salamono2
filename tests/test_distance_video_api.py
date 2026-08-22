import asyncio
import base64
from io import BytesIO
import threading

import cv2
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import numpy as np
import pytest

from backend.demo_video import DemoVideoService
from backend.detector import Detection
from backend.distance_measurement import DistancePair, GroundPlaneDistanceService
from backend.routes import distance
from config import DemoVideoConfig


class _SeekCapture:
    def __init__(self, *, width=160, height=90, fps=10.0, frame_count=20):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.frame_count = int(frame_count)
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.position >= self.frame_count:
            return False, None
        value = self.position % 255
        frame = np.full((self.height, self.width, 3), value, dtype=np.uint8)
        self.position += 1
        return True, frame

    def grab(self):
        if self.position >= self.frame_count:
            return False
        self.position += 1
        return True

    def set(self, prop_id, value):
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self.position = max(0, int(round(float(value))))
            return True
        if prop_id == cv2.CAP_PROP_POS_MSEC:
            self.position = max(0, int(round(float(value) * self.fps / 1000.0)))
            return True
        return False

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.frame_count
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return self.position
        if prop_id == cv2.CAP_PROP_POS_MSEC:
            return max(0, self.position - 1) * 1000.0 / self.fps
        return 0.0

    def release(self):
        self.released = True


class _Detector:
    def __init__(self, detections, lock):
        self.detections = list(detections)
        self.lock = lock
        self.frame_values = []

    def detect(self, frame):
        assert self.lock.locked(), "manual inference must use app.state.analysis_lock"
        self.frame_values.append(int(round(float(frame.mean()))))
        return list(self.detections)


class _DistanceService:
    def __init__(self, *, compatible=True, available=True, distance_m=2.0):
        self.compatible = compatible
        self.available = available
        self.error = None if available else "missing calibration"
        self.distance_m = float(distance_m)
        self.measure_calls = 0

    def status(self, width=None, height=None):
        warning = None
        if self.available and width is not None and not self.compatible:
            warning = "Niezgodny aspect ratio kalibracji i klipu."
        return {
            "available": self.available,
            "error": self.error,
            "compatible": (
                self.compatible if self.available and width is not None else None
            ),
            "warning": warning,
            "calibration_width": 160,
            "calibration_height": 90,
        }

    def measure(self, detections, width, height, *, max_distance_m=None):
        self.measure_calls += 1
        persons = [item for item in detections if item.category == "person"]
        hazards = [item for item in detections if item.category == "vehicle"]
        if not persons or not hazards:
            return [], len(persons), len(hazards)
        if max_distance_m is not None and self.distance_m > max_distance_m:
            return [], len(persons), len(hazards)
        person = persons[0]
        hazard = hazards[0]
        pair = DistancePair(
            person_index=0,
            hazard_index=0,
            person_class=person.class_name,
            hazard_class=hazard.class_name,
            person_confidence=person.confidence,
            hazard_confidence=hazard.confidence,
            person_box=person.box,
            hazard_box=hazard.box,
            distance_m=self.distance_m,
            person_point_px=(40, 80),
            hazard_point_px=(100, 80),
        )
        return [pair], len(persons), len(hazards)


def _detections():
    return [
        Detection(0, "person", "person", (20, 15, 55, 80), 0.91),
        Detection(3, "excavator", "vehicle", (90, 20, 145, 80), 0.87),
    ]


def _service(tmp_path, *, capture_factory=None, max_pending_jobs=2, process_frame=None):
    config = DemoVideoConfig(
        upload_dir=str(tmp_path / "uploads"),
        output_dir=str(tmp_path / "outputs"),
        max_size_mb=8,
        job_ttl_seconds=60,
        max_pending_jobs=max_pending_jobs,
        processing_enabled=True,
        playback_mode="fast",
        probe_timeout_seconds=0.5,
        control_timeout_seconds=1.0,
    )
    return DemoVideoService(
        config,
        process_frame=process_frame,
        capture_factory=capture_factory or (lambda _path: _SeekCapture()),
        start_janitor=False,
    )


def _app(service, detector, lock):
    app = FastAPI()
    app.state.demo_video_service = service
    app.state.detector = detector
    app.state.analysis_lock = lock
    app.include_router(distance.router, prefix="/api")
    return app


def test_ground_plane_calibration_scales_same_aspect_and_rejects_mismatch():
    service = GroundPlaneDistanceService()
    assert service.status(1280, 720)["compatible"] is True
    mismatch = service.status(1024, 768)
    assert mismatch["compatible"] is False
    assert "aspect ratio" in mismatch["warning"]


@pytest.mark.asyncio
async def test_distance_clip_upload_seek_reanalyze_content_and_explicit_delete(
    tmp_path, monkeypatch
):
    service = _service(tmp_path)
    analysis_lock = threading.Lock()
    detector = _Detector(_detections(), analysis_lock)
    distance_service = _DistanceService(distance_m=2.0)
    monkeypatch.setattr(distance, "_service", distance_service)
    app = _app(service, detector, analysis_lock)
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            upload = await client.post(
                "/api/distance/videos",
                files={"video": ("shift.mp4", b"0123456789", "video/mp4")},
            )
            assert upload.status_code == 200
            metadata = upload.json()
            video_id = metadata["id"]
            assert metadata == {
                **metadata,
                "job_id": video_id,
                "filename": "shift.mp4",
                "width": 160,
                "height": 90,
                "fps": 10.0,
                "frame_count": 20,
                "duration_sec": 2.0,
                "status": "ready",
                "video_url": f"/api/distance/videos/{video_id}/content",
            }
            assert service.get(video_id).camera_id.startswith("distance_clip:")
            assert service.asset_kind_for(video_id) == "distance"

            fetched = await client.get(f"/api/distance/videos/{video_id}")
            assert fetched.status_code == 200
            assert fetched.json()["frame_count"] == 20

            ranged = await client.get(
                f"/api/distance/videos/{video_id}/content",
                headers={"Range": "bytes=2-5"},
            )
            assert ranged.status_code == 206
            assert ranged.content == b"2345"
            assert ranged.headers["accept-ranges"] == "bytes"
            with service._condition:
                assert service._jobs[video_id].read_only_leases == 0

            frame = await client.post(
                f"/api/distance/videos/{video_id}/frame",
                json={"time_sec": 0.21, "max_width": 1280},
            )
            assert frame.status_code == 200
            frame_data = frame.json()
            assert frame_data["requested_time_sec"] == pytest.approx(0.21)
            assert frame_data["requested_frame_index"] == 2
            assert frame_data["frame_index"] == 2
            assert frame_data["actual_time_sec"] == pytest.approx(0.2)
            preview = cv2.imdecode(
                np.frombuffer(base64.b64decode(frame_data["preview_jpeg_b64"]), np.uint8),
                cv2.IMREAD_COLOR,
            )
            assert preview.shape[:2] == (90, 160)
            assert float(preview.mean()) == pytest.approx(2.0, abs=2.0)

            first = await client.post(
                f"/api/distance/videos/{video_id}/analyze",
                json={"time_sec": 0.41},
            )
            assert first.status_code == 200
            first_data = first.json()
            assert first_data["frame_index"] == 4
            assert first_data["actual_time_sec"] == pytest.approx(0.4)
            assert first_data["status"] == "warning"
            assert first_data["pair_count"] == 1
            assert first_data["pairs"][0]["status"] == "warning"
            assert first_data["pairs"][0]["hazard_class"] == "excavator"
            assert {item["category"] for item in first_data["detections"]} == {
                "person",
                "vehicle",
            }

            second = await client.post(
                f"/api/distance/videos/{video_id}/analyze",
                json={"frame_index": 7},
            )
            assert second.status_code == 200
            assert second.json()["frame_index"] == 7
            assert detector.frame_values == [4, 7]
            assert distance_service.measure_calls == 2
            assert len(service.list_jobs()) == 1

            deleted = await client.delete(f"/api/distance/videos/{video_id}")
            assert deleted.status_code == 204
            assert (await client.get(f"/api/distance/videos/{video_id}")).status_code == 404
            assert service.list_jobs() == []
    finally:
        service.close(timeout=1.0)


@pytest.mark.asyncio
async def test_time_seek_uses_pos_msec_for_vfr_and_frame_seek_uses_pos_frames(
    tmp_path, monkeypatch
):
    captures = []

    class VfrCapture(_SeekCapture):
        def __init__(self):
            super().__init__(fps=10.0, frame_count=20)
            self.set_calls = []
            self.actual_msec = 0.0

        def set(self, prop_id, value):
            self.set_calls.append((prop_id, float(value)))
            if prop_id == cv2.CAP_PROP_POS_MSEC:
                # Average-FPS math would choose frame 4, but this VFR timestamp
                # resolves to source frame 6 at an actual timestamp of 450 ms.
                self.position = 6
                self.actual_msec = 450.0
                return True
            if prop_id == cv2.CAP_PROP_POS_FRAMES:
                self.position = int(round(float(value)))
                self.actual_msec = self.position * 100.0
                return True
            return False

        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_POS_MSEC:
                return self.actual_msec
            return super().get(prop_id)

    def capture_factory(_path):
        # First capture is the cheap metadata probe; later captures record seek.
        capture = _SeekCapture() if not captures else VfrCapture()
        captures.append(capture)
        return capture

    service = _service(tmp_path, capture_factory=capture_factory)
    analysis_lock = threading.Lock()
    detector = _Detector([], analysis_lock)
    monkeypatch.setattr(distance, "_service", _DistanceService())
    app = _app(service, detector, analysis_lock)
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            upload = await client.post(
                "/api/distance/videos",
                files={"video": ("vfr.mp4", b"video", "video/mp4")},
            )
            video_id = upload.json()["id"]

            by_time = await client.post(
                f"/api/distance/videos/{video_id}/analyze",
                json={"time_sec": 0.41},
            )
            assert by_time.status_code == 200
            time_data = by_time.json()
            assert time_data["seek_mode"] == "time"
            assert time_data["requested_frame_index"] == 4
            assert time_data["frame_index"] == 6
            assert time_data["actual_time_sec"] == pytest.approx(0.45)
            assert captures[1].set_calls[0][0] == cv2.CAP_PROP_POS_MSEC

            by_frame = await client.post(
                f"/api/distance/videos/{video_id}/analyze",
                json={"frame_index": 2},
            )
            assert by_frame.status_code == 200
            assert by_frame.json()["seek_mode"] == "frame"
            assert by_frame.json()["frame_index"] == 2
            assert captures[2].set_calls[0][0] == cv2.CAP_PROP_POS_FRAMES
    finally:
        service.close(timeout=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("detections", "compatible", "expected_status", "measure_calls"),
    [
        ([_detections()[0]], True, "no_pairs", 1),
        (_detections(), False, "calibration_incompatible", 0),
    ],
)
async def test_distance_clip_empty_pairs_and_incompatible_calibration_fail_closed(
    tmp_path,
    monkeypatch,
    detections,
    compatible,
    expected_status,
    measure_calls,
):
    service = _service(tmp_path)
    analysis_lock = threading.Lock()
    detector = _Detector(detections, analysis_lock)
    distance_service = _DistanceService(compatible=compatible)
    monkeypatch.setattr(distance, "_service", distance_service)
    app = _app(service, detector, analysis_lock)
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            uploaded = await client.post(
                "/api/distance/videos",
                files={"video": ("clip.mp4", b"video", "video/mp4")},
            )
            video_id = uploaded.json()["id"]
            analyzed = await client.post(
                f"/api/distance/videos/{video_id}/analyze",
                json={"frame_index": 3},
            )
            assert analyzed.status_code == 200
            body = analyzed.json()
            assert body["status"] == expected_status
            assert body["pairs"] == []
            assert body["pair_count"] == 0
            assert distance_service.measure_calls == measure_calls
            if not compatible:
                assert "aspect ratio" in body["message"].lower()
    finally:
        service.close(timeout=1.0)


@pytest.mark.asyncio
async def test_distance_clip_4k_is_local_and_preview_is_scaled(tmp_path, monkeypatch):
    service = _service(
        tmp_path,
        capture_factory=lambda _path: _SeekCapture(
            width=3840,
            height=2160,
            fps=29.97,
            frame_count=3,
        ),
    )
    analysis_lock = threading.Lock()
    detector = _Detector([], analysis_lock)
    monkeypatch.setattr(distance, "_service", _DistanceService())
    app = _app(service, detector, analysis_lock)
    # Deliberately do not install LatestFrameStore: a selected 4K frame stays
    # request-local and cannot hit its decoded-frame byte limit.
    assert not hasattr(app.state, "latest_frame_store")
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            upload = await client.post(
                "/api/distance/videos",
                files={"video": ("4k.mov", b"video", "video/quicktime")},
            )
            assert upload.status_code == 200
            assert upload.json()["width"] == 3840
            video_id = upload.json()["id"]
            analyzed = await client.post(
                f"/api/distance/videos/{video_id}/analyze",
                json={"frame_index": 1, "preview_max_width": 1280},
            )
            assert analyzed.status_code == 200
            body = analyzed.json()
            assert (body["width"], body["height"]) == (3840, 2160)
            assert (body["preview_width"], body["preview_height"]) == (1280, 720)
            assert body["preview_scale"] == pytest.approx(1.0 / 3.0)
            assert body["pairs"] == []
    finally:
        service.close(timeout=1.0)


@pytest.mark.asyncio
async def test_distance_routes_cannot_read_or_delete_regular_demo_job(
    tmp_path, monkeypatch
):
    service = _service(tmp_path)
    demo = service.upload_stream(
        "demo.mp4",
        "video/mp4",
        BytesIO(b"demo"),
        # camera_id is public/spoofable and must not grant distance ownership.
        camera_id="distance_clip:spoofed",
    )
    analysis_lock = threading.Lock()
    detector = _Detector([], analysis_lock)
    monkeypatch.setattr(distance, "_service", _DistanceService())
    app = _app(service, detector, analysis_lock)
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            assert (
                await client.get(f"/api/distance/videos/{demo.job_id}")
            ).status_code == 404
            assert (
                await client.delete(f"/api/distance/videos/{demo.job_id}")
            ).status_code == 404
            assert service.get(demo.job_id).status == "uploaded"
    finally:
        service.close(timeout=1.0)


@pytest.mark.asyncio
async def test_read_only_clip_probe_does_not_compete_with_active_playback_slot(
    tmp_path, monkeypatch
):
    playback_entered = threading.Event()
    release_playback = threading.Event()

    def process_frame(_context, _frame):
        playback_entered.set()
        release_playback.wait(timeout=2.0)

    service = _service(
        tmp_path,
        max_pending_jobs=1,
        process_frame=process_frame,
        capture_factory=lambda _path: _SeekCapture(fps=30.0, frame_count=100),
    )
    demo = service.upload_stream(
        "demo.mp4",
        "video/mp4",
        BytesIO(b"demo"),
        camera_id="demo_upload",
        playback_mode="realtime",
    )
    service.start_probe(demo.job_id)
    service.wait_for_status(demo.job_id, "ready", timeout=1.0)
    service.play(demo.job_id)
    assert playback_entered.wait(timeout=1.0)

    analysis_lock = threading.Lock()
    detector = _Detector([], analysis_lock)
    monkeypatch.setattr(distance, "_service", _DistanceService())
    app = _app(service, detector, analysis_lock)
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            upload = await client.post(
                "/api/distance/videos",
                files={"video": ("manual.mp4", b"clip", "video/mp4")},
            )
            assert upload.status_code == 200
            assert upload.json()["status"] == "ready"
            assert service.get(demo.job_id).status == "playing"
    finally:
        release_playback.set()
        service.close(timeout=2.0)


@pytest.mark.asyncio
async def test_parallel_distance_upload_probe_returns_409_without_orphan_job(
    tmp_path, monkeypatch
):
    probe_entered = threading.Event()
    release_probe = threading.Event()

    class BlockingProbeCapture(_SeekCapture):
        def read(self):
            probe_entered.set()
            assert release_probe.wait(1.0)
            return super().read()

    calls = [0]

    def capture_factory(_path):
        calls[0] += 1
        return BlockingProbeCapture() if calls[0] == 1 else _SeekCapture()

    service = _service(tmp_path, capture_factory=capture_factory)
    analysis_lock = threading.Lock()
    detector = _Detector([], analysis_lock)
    monkeypatch.setattr(distance, "_service", _DistanceService())
    app = _app(service, detector, analysis_lock)
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            first_task = asyncio.create_task(
                client.post(
                    "/api/distance/videos",
                    files={"video": ("first.mp4", b"first", "video/mp4")},
                )
            )
            assert await asyncio.to_thread(probe_entered.wait, 0.5)

            second = await client.post(
                "/api/distance/videos",
                files={"video": ("second.mp4", b"second", "video/mp4")},
            )
            assert second.status_code == 409
            assert "probe capacity" in second.json()["detail"]
            jobs = service.list_jobs()
            assert len(jobs) == 1
            assert jobs[0].filename == "first.mp4"

            release_probe.set()
            first = await first_task
            assert first.status_code == 200
    finally:
        release_probe.set()
        service.close(timeout=2.0)


@pytest.mark.asyncio
async def test_manual_slot_stays_held_after_cancel_until_worker_finishes(
    tmp_path, monkeypatch
):
    decode_entered = threading.Event()
    release_decode = threading.Event()
    inference_finished = threading.Event()
    calls = [0]

    class BlockingManualCapture(_SeekCapture):
        def read(self):
            decode_entered.set()
            assert release_decode.wait(1.0)
            return super().read()

    def capture_factory(_path):
        calls[0] += 1
        # Upload metadata probe remains cheap; manual requests block on demand.
        return _SeekCapture() if calls[0] == 1 else BlockingManualCapture()

    service = _service(tmp_path, capture_factory=capture_factory)
    analysis_lock = threading.Lock()

    class SignallingDetector(_Detector):
        def detect(self, frame):
            result = super().detect(frame)
            inference_finished.set()
            return result

    detector = SignallingDetector([], analysis_lock)
    monkeypatch.setattr(distance, "_service", _DistanceService())
    app = _app(service, detector, analysis_lock)
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            upload = await client.post(
                "/api/distance/videos",
                files={"video": ("manual.mp4", b"video", "video/mp4")},
            )
            video_id = upload.json()["id"]
            first = asyncio.create_task(
                client.post(
                    f"/api/distance/videos/{video_id}/analyze",
                    json={"time_sec": 0.2},
                )
            )
            assert await asyncio.to_thread(decode_entered.wait, 0.5)
            with service._condition:
                assert service._jobs[video_id].read_only_leases == 1

            busy = await client.post(
                f"/api/distance/videos/{video_id}/frame",
                json={"time_sec": 0.3},
            )
            assert busy.status_code == 429

            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            # Cancellation of the HTTP coroutine must not release either the
            # manual slot or the decoder's source lease while to_thread runs.
            still_busy = await client.post(
                f"/api/distance/videos/{video_id}/frame",
                json={"time_sec": 0.4},
            )
            assert still_busy.status_code == 429
            with service._condition:
                assert service._jobs[video_id].read_only_leases == 1

            release_decode.set()
            assert await asyncio.to_thread(inference_finished.wait, 0.5)
            accepted = None
            for _ in range(30):
                accepted = await client.post(
                    f"/api/distance/videos/{video_id}/frame",
                    json={"time_sec": 0.5},
                )
                if accepted.status_code != 429:
                    break
                await asyncio.sleep(0.01)
            assert accepted is not None and accepted.status_code == 200
            with service._condition:
                assert service._jobs[video_id].read_only_leases == 0
    finally:
        release_decode.set()
        service.close(timeout=2.0)
