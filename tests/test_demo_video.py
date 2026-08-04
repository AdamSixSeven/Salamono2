from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import pytest

from backend.demo_video import (
    DemoVideoConflictError,
    DemoVideoNotFoundError,
    DemoVideoService,
    DemoVideoStatus,
    DemoVideoTimeoutError,
    DemoVideoTooLargeError,
    DemoVideoValidationError,
)
from config import DemoVideoConfig


class FakeCapture:
    def __init__(
        self,
        frames,
        *,
        fps=10.0,
        opened=True,
        read_delay=0.0,
    ):
        self.frames = [np.array(frame, copy=True) for frame in frames]
        self.fps = float(fps)
        self.opened = bool(opened)
        self.read_delay = float(read_delay)
        self.position = 0
        self.release_calls = 0

    def isOpened(self):
        return self.opened

    def read(self):
        if self.read_delay:
            time.sleep(self.read_delay)
        if self.position >= len(self.frames):
            return False, None
        frame = np.array(self.frames[self.position], copy=True)
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
            return self.frames[0].shape[1] if self.frames else 0
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.frames[0].shape[0] if self.frames else 0
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return self.position
        if prop_id == cv2.CAP_PROP_POS_MSEC:
            return (self.position / max(self.fps, 0.1)) * 1000.0
        return 0

    def release(self):
        self.release_calls += 1


class CaptureFactory:
    def __init__(self, frames, **capture_options):
        self.frames = list(frames)
        self.capture_options = capture_options
        self.instances = []
        self.lock = threading.Lock()

    def __call__(self, _path):
        capture = FakeCapture(self.frames, **self.capture_options)
        with self.lock:
            self.instances.append(capture)
        return capture


def frame(value=0):
    return np.full((12, 16, 3), value, dtype=np.uint8)


def config(tmp_path, **changes):
    base = DemoVideoConfig(
        upload_dir=str(tmp_path / "uploads"),
        output_dir=str(tmp_path / "outputs"),
        max_size_mb=1,
        allowed_extensions=(".mp4", ".avi", ".mov", ".mkv"),
        job_ttl_seconds=60.0,
        max_pending_jobs=2,
        processing_enabled=True,
        playback_mode="fast",
        probe_timeout_seconds=0.25,
        control_timeout_seconds=0.5,
    )
    return replace(base, **changes)


def upload(service, *, name="clip.mp4", mime="video/mp4", camera_id="demo_upload"):
    snapshot = service.upload_stream(
        name,
        mime,
        BytesIO(b"fake-video-payload"),
        camera_id=camera_id,
    )
    assert snapshot.status == "uploaded"
    return snapshot


def ready(service, **upload_kwargs):
    snapshot = upload(service, **upload_kwargs)
    loading = service.start_probe(snapshot.job_id)
    assert loading.status == "loading"
    return service.wait_for_status(
        snapshot.job_id,
        {DemoVideoStatus.READY, DemoVideoStatus.FAILED},
        timeout=1.0,
    )


def close(service):
    try:
        service.close(timeout=1.0)
    except DemoVideoTimeoutError:
        # A timeout test can leave its deliberately slow daemon helper alive
        # for a few more milliseconds; the job worker itself has still ended.
        pass


def test_streaming_upload_sanitizes_path_and_records_metrics(tmp_path):
    factory = CaptureFactory([frame()])
    service = DemoVideoService(
        config(tmp_path),
        capture_factory=factory,
        start_janitor=False,
    )
    try:
        created = service.create_upload("../../unsafe clip.mp4", "video/mp4")
        assert created.status == "uploading"
        assert created.filename == "unsafe clip.mp4"
        stored = service.write_upload(created.job_id, BytesIO(b"abcdefgh"), chunk_size=3)

        path = Path(stored.input_path).resolve()
        assert path.read_bytes() == b"abcdefgh"
        assert path.parent.parent == Path(service.upload_dir)
        assert stored.status == "uploaded"
        assert stored.size_bytes == 8
        metrics = service.stats()
        assert metrics["demo_upload_size_bytes"] == 8
        assert set(metrics) == {
            "demo_upload_ms",
            "demo_upload_size_bytes",
            "demo_probe_ms",
            "demo_decode_ms",
            "demo_processing_ms",
            "demo_source_fps",
            "demo_processing_fps",
            "demo_dropped_frames",
            "demo_job_queue_age_ms",
            "demo_active_jobs",
        }
    finally:
        close(service)


@pytest.mark.parametrize(
    ("name", "mime"),
    [
        ("clip.exe", "video/mp4"),
        ("clip.mp4", "text/plain"),
        ("clip", "video/mp4"),
        ("..", "video/mp4"),
    ],
)
def test_upload_rejects_invalid_extension_mime_and_name(tmp_path, name, mime):
    service = DemoVideoService(config(tmp_path), start_janitor=False)
    try:
        with pytest.raises(DemoVideoValidationError):
            service.create_upload(name, mime)
        assert service.list_jobs() == []
    finally:
        close(service)


def test_upload_limit_removes_partial_file_and_marks_failed(tmp_path):
    service = DemoVideoService(
        config(tmp_path),
        max_size_bytes=5,
        start_janitor=False,
    )
    try:
        created = service.create_upload("clip.mp4", "video/mp4")
        with pytest.raises(DemoVideoTooLargeError):
            service.write_upload(created.job_id, BytesIO(b"123456"), chunk_size=2)
        failed = service.get(created.job_id)
        assert failed.status == "failed"
        assert "exceeds" in (failed.error or "")
        assert not Path(failed.input_path).exists()
    finally:
        close(service)


def test_probe_populates_metadata_and_always_releases_capture(tmp_path):
    factory = CaptureFactory([frame(1), frame(2), frame(3)], fps=12.0)
    service = DemoVideoService(
        config(tmp_path),
        capture_factory=factory,
        start_janitor=False,
    )
    try:
        snapshot = ready(service)
        assert snapshot.status == "ready"
        assert snapshot.total_frames == 3
        assert snapshot.source_fps == 12.0
        assert snapshot.duration_sec == pytest.approx(0.25)
        assert (snapshot.width, snapshot.height) == (16, 12)
        assert factory.instances[0].release_calls == 1
        assert service.stats()["demo_probe_ms"] >= 0
    finally:
        close(service)


def test_corrupt_video_reaches_failed_instead_of_staying_loading(tmp_path):
    factory = CaptureFactory([], fps=10.0)
    service = DemoVideoService(
        config(tmp_path),
        capture_factory=factory,
        start_janitor=False,
    )
    try:
        snapshot = ready(service)
        assert snapshot.status == "failed"
        assert "no decodable frames" in (snapshot.error or "")
        assert factory.instances[0].release_calls == 1
    finally:
        close(service)


def test_probe_watchdog_fails_and_releases_slow_capture(tmp_path):
    factory = CaptureFactory([frame()], read_delay=0.15)
    service = DemoVideoService(
        config(tmp_path, probe_timeout_seconds=0.03),
        capture_factory=factory,
        start_janitor=False,
    )
    try:
        uploaded = upload(service)
        service.start_probe(uploaded.job_id)
        snapshot = service.wait_for_status(
            uploaded.job_id,
            DemoVideoStatus.FAILED,
            timeout=0.2,
        )
        assert "exceeded" in (snapshot.error or "")
        assert factory.instances[0].release_calls == 1
    finally:
        time.sleep(0.16)
        close(service)


def test_stop_during_loading_preserves_probe_metadata_for_later_play(tmp_path):
    factory = CaptureFactory([frame(1), frame(2)], read_delay=0.05)
    service = DemoVideoService(
        config(tmp_path, control_timeout_seconds=0.5),
        capture_factory=factory,
        start_janitor=False,
    )
    try:
        uploaded = upload(service)
        service.start_probe(uploaded.job_id)
        stopped = service.stop(uploaded.job_id, timeout=0.5)

        assert stopped.status == "stopped"
        assert stopped.width == 16
        assert stopped.height == 12
        assert stopped.total_frames == 2

        assert service.play(uploaded.job_id).status in {"playing", "finished"}
        service.wait_for_status(uploaded.job_id, "finished", timeout=1.0)
    finally:
        close(service)


def test_play_finishes_and_restart_uses_new_run_from_frame_zero(tmp_path):
    factory = CaptureFactory([frame(1), frame(2), frame(3)], fps=20.0)
    contexts = []
    resets = []
    service = DemoVideoService(
        config(tmp_path),
        capture_factory=factory,
        process_frame=lambda context, _frame: contexts.append(context),
        reset_camera=resets.append,
        start_janitor=False,
    )
    try:
        job = ready(service)
        first_play = service.play(job.job_id)
        first_run = first_play.run_id
        finished = service.wait_for_status(job.job_id, "finished", timeout=1.0)
        assert finished.current_frame == 3
        assert [item.frame_index for item in contexts] == [0, 1, 2]
        assert {item.run_id for item in contexts} == {first_run}

        replay = service.restart(job.job_id, autoplay=True)
        assert replay.run_id != first_run
        second_run = replay.run_id
        service.wait_for_status(job.job_id, "finished", timeout=1.0)
        assert [item.frame_index for item in contexts] == [0, 1, 2, 0, 1, 2]
        assert [item.run_id for item in contexts[-3:]] == [second_run] * 3
        assert resets == ["demo_upload", "demo_upload"]
        assert all(capture.release_calls == 1 for capture in factory.instances)
    finally:
        close(service)


def test_restart_without_autoplay_resets_position_and_prepares_run(tmp_path):
    factory = CaptureFactory([frame(1), frame(2)], fps=10.0)
    service = DemoVideoService(
        config(tmp_path),
        capture_factory=factory,
        start_janitor=False,
    )
    try:
        job = ready(service)
        service.play(job.job_id)
        service.wait_for_status(job.job_id, "finished", timeout=1.0)
        prepared = service.restart(job.job_id, autoplay=False)
        assert prepared.status == "ready"
        assert prepared.current_frame == 0
        assert prepared.current_time_sec == 0
        assert prepared.run_id
        run_id = prepared.run_id
        playing = service.play(job.job_id)
        assert playing.run_id == run_id
    finally:
        close(service)


def test_pause_waits_for_current_frame_and_resume_keeps_position(tmp_path):
    factory = CaptureFactory([frame(1), frame(2), frame(3)], fps=10.0)
    first_started = threading.Event()
    release_first = threading.Event()
    contexts = []

    def process(context, _frame):
        contexts.append(context)
        if context.frame_index == 0:
            first_started.set()
            assert release_first.wait(1.0)

    service = DemoVideoService(
        config(tmp_path),
        capture_factory=factory,
        process_frame=process,
        start_janitor=False,
    )
    try:
        job = ready(service)
        service.play(job.job_id)
        assert first_started.wait(0.5)
        requested = service.pause(job.job_id, wait=False)
        assert requested.pause_requested is True
        assert requested.current_frame == 0
        release_first.set()
        paused = service.wait_for_status(job.job_id, "paused", timeout=1.0)
        assert paused.current_frame == 1
        position = paused.current_frame
        resumed = service.resume(job.job_id)
        assert resumed.status == "playing"
        service.wait_for_status(job.job_id, "finished", timeout=1.0)
        assert position == 1
        assert [item.frame_index for item in contexts] == [0, 1, 2]
    finally:
        release_first.set()
        close(service)


def test_stop_keeps_upload_and_delete_releases_and_removes_job(tmp_path):
    factory = CaptureFactory([frame(index) for index in range(20)], fps=10.0)
    processed = threading.Event()

    def process(_context, _frame):
        processed.set()

    service = DemoVideoService(
        config(tmp_path, playback_mode="realtime"),
        capture_factory=factory,
        process_frame=process,
        start_janitor=False,
    )
    try:
        job = ready(service)
        service.play(job.job_id)
        assert processed.wait(0.5)
        stopped = service.stop(job.job_id, timeout=1.0)
        assert stopped.status == "stopped"
        assert Path(stopped.input_path).is_file()
        assert factory.instances[-1].release_calls == 1

        deleted = service.delete(job.job_id)
        assert deleted.status == "deleted"
        assert not Path(deleted.input_path).parent.exists()
        with pytest.raises(DemoVideoNotFoundError):
            service.get(job.job_id)
    finally:
        close(service)


def test_realtime_mode_drops_late_frames_without_processing_backlog(tmp_path):
    frames = [frame(index) for index in range(30)]
    factory = CaptureFactory(frames, fps=100.0)
    contexts = []

    def slow_process(context, _frame):
        contexts.append(context)
        time.sleep(0.035)

    service = DemoVideoService(
        config(tmp_path, playback_mode="realtime"),
        capture_factory=factory,
        process_frame=slow_process,
        start_janitor=False,
    )
    try:
        job = ready(service)
        started = time.monotonic()
        service.play(job.job_id)
        finished = service.wait_for_status(job.job_id, "finished", timeout=1.5)
        elapsed = time.monotonic() - started
        assert finished.dropped_frames > 0
        assert len(contexts) + finished.dropped_frames == 30
        assert len(contexts) < 15
        assert elapsed < 0.8
        assert service.stats()["demo_dropped_frames"] == finished.dropped_frames
    finally:
        close(service)


def test_fast_mode_processes_every_frame(tmp_path):
    factory = CaptureFactory([frame(index) for index in range(8)], fps=100.0)
    indices = []
    service = DemoVideoService(
        config(tmp_path, playback_mode="fast"),
        capture_factory=factory,
        process_frame=lambda context, _frame: indices.append(context.frame_index),
        start_janitor=False,
    )
    try:
        job = ready(service)
        service.play(job.job_id)
        finished = service.wait_for_status(job.job_id, "finished", timeout=1.0)
        assert indices == list(range(8))
        assert finished.dropped_frames == 0
    finally:
        close(service)


def test_real_opencv_avi_is_uploaded_probed_and_processed(tmp_path):
    source_path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        8.0,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV build has no MJPG VideoWriter")
    for value in (20, 80, 140):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()

    contexts = []
    service = DemoVideoService(
        config(tmp_path / "service", playback_mode="fast"),
        process_frame=lambda context, _frame: contexts.append(context),
        start_janitor=False,
    )
    try:
        uploaded = service.upload_stream(
            "source.avi",
            "video/x-msvideo",
            BytesIO(source_path.read_bytes()),
        )
        service.start_probe(uploaded.job_id)
        prepared = service.wait_for_status(
            uploaded.job_id,
            {"ready", "failed"},
            timeout=2.0,
        )
        assert prepared.status == "ready", prepared.error
        assert prepared.total_frames == 3

        service.play(uploaded.job_id)
        finished = service.wait_for_status(
            uploaded.job_id,
            "finished",
            timeout=2.0,
        )
        assert finished.current_frame == 3
        assert [context.frame_index for context in contexts] == [0, 1, 2]
    finally:
        close(service)


def test_export_processes_every_frame_and_writes_annotated_mp4(tmp_path):
    source_path = tmp_path / "export-source.avi"
    writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        8.0,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV build has no MJPG VideoWriter")
    for value in (20, 80, 140):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()

    contexts = []

    def render(context, source_frame):
        contexts.append(context)
        rendered = source_frame.copy()
        rendered[:, :4] = (0, 255, 0)
        return rendered

    service = DemoVideoService(
        config(tmp_path / "service", playback_mode="realtime"),
        process_frame=render,
        start_janitor=False,
    )
    try:
        uploaded = service.upload_stream(
            "export-source.avi",
            "video/x-msvideo",
            BytesIO(source_path.read_bytes()),
        )
        service.start_probe(uploaded.job_id)
        prepared = service.wait_for_status(uploaded.job_id, "ready", timeout=2.0)
        assert prepared.status == "ready"

        started = service.export(uploaded.job_id)
        assert started.playback_mode == "fast"
        assert started.export_status == "processing"
        service.wait_for_status(uploaded.job_id, "finished", timeout=2.0)
        deadline = time.time() + 1.0
        while service.get(uploaded.job_id).export_status != "ready" and time.time() < deadline:
            time.sleep(0.01)
        finished = service.get(uploaded.job_id)
        assert finished.export_status == "ready"
        assert all(context.export_mode for context in contexts)
        assert [context.frame_index for context in contexts] == [0, 1, 2]

        output_path = service.output_path_for(uploaded.job_id)
        assert output_path.name == "export-source_perimetr_ai.mp4"
        assert output_path.stat().st_size > 0
        output = cv2.VideoCapture(str(output_path))
        try:
            assert output.isOpened()
            assert int(output.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
            fourcc_value = int(output.get(cv2.CAP_PROP_FOURCC))
            fourcc = "".join(
                chr((fourcc_value >> (8 * index)) & 0xFF)
                for index in range(4)
            ).lower()
            assert fourcc in {"avc1", "h264"}
        finally:
            output.release()
    finally:
        close(service)


def test_only_one_active_job_may_use_camera_id(tmp_path):
    factory = CaptureFactory([frame(index) for index in range(20)], fps=2.0)
    service = DemoVideoService(
        config(tmp_path, playback_mode="realtime"),
        capture_factory=factory,
        start_janitor=False,
    )
    try:
        first = ready(service)
        second = ready(service)
        service.play(first.job_id)
        with pytest.raises(DemoVideoConflictError):
            service.play(second.job_id)
        service.stop(first.job_id, timeout=1.0)
        assert service.play(second.job_id).status == "playing"
    finally:
        close(service)


def test_shutdown_stops_active_worker_and_releases_capture(tmp_path):
    factory = CaptureFactory([frame(index) for index in range(20)], fps=1.0)
    service = DemoVideoService(
        config(tmp_path, playback_mode="realtime"),
        capture_factory=factory,
        start_janitor=False,
    )
    job = ready(service)
    upload_path = Path(job.input_path)
    service.play(job.job_id)
    service.close(timeout=1.0)
    snapshot = service.get(job.job_id)
    assert snapshot.status == "stopped"
    assert factory.instances[-1].release_calls == 1
    assert not upload_path.parent.exists()
    with pytest.raises(DemoVideoConflictError):
        service.create_upload("another.mp4", "video/mp4")


def test_ttl_cleanup_removes_inactive_job_and_files(tmp_path):
    now = [1000.0]
    service = DemoVideoService(
        config(tmp_path, job_ttl_seconds=10.0),
        clock=lambda: now[0],
        start_janitor=False,
    )
    try:
        job = upload(service)
        path = Path(job.input_path)
        now[0] += 11.0
        assert service.cleanup_expired() == 1
        assert not path.parent.exists()
        with pytest.raises(DemoVideoNotFoundError):
            service.get(job.job_id)
    finally:
        close(service)


def test_startup_cleanup_removes_only_old_orphan_directories(tmp_path):
    upload_root = tmp_path / "uploads"
    output_root = tmp_path / "outputs"
    old = upload_root / "old-job"
    recent = output_root / "recent-job"
    old.mkdir(parents=True)
    recent.mkdir(parents=True)
    (old / "clip.mp4").write_bytes(b"old")
    old_time = time.time() - 100
    os.utime(old, (old_time, old_time))

    service = DemoVideoService(
        config(tmp_path, job_ttl_seconds=10.0),
        start_janitor=False,
    )
    try:
        assert not old.exists()
        assert recent.exists()
    finally:
        close(service)


def test_invalid_state_actions_are_rejected(tmp_path):
    service = DemoVideoService(config(tmp_path), start_janitor=False)
    try:
        job = upload(service)
        with pytest.raises(DemoVideoConflictError):
            service.play(job.job_id)
        with pytest.raises(DemoVideoConflictError):
            service.pause(job.job_id)
        with pytest.raises(DemoVideoConflictError):
            service.resume(job.job_id)
    finally:
        close(service)
