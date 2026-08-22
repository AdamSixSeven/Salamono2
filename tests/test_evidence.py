import os
import sys
import threading

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.evidence import EvidenceRecorder
from config import EvidenceConfig


def test_evidence_recorder_writes_short_mp4(tmp_path):
    cfg = EvidenceConfig(
        enabled=True,
        clips_dir=str(tmp_path),
        pre_seconds=1.0,
        post_seconds=0.0,
        sample_fps=5.0,
        max_buffer_frames=10,
    )
    recorder = EvidenceRecorder(cfg)
    for idx in range(4):
        frame = np.full((64, 96, 3), idx * 40, dtype=np.uint8)
        recorder.push("cam", frame, 10.0 + idx * 0.25)
    url = recorder.trigger("cam", "abc123", 10.75)
    assert url is not None
    # close() is a durability boundary: it drains the background writer.
    recorder.close()
    filename = url.rsplit("/", 1)[-1]
    path = tmp_path / filename
    assert path.exists()
    assert path.stat().st_size > 0
    capture = cv2.VideoCapture(str(path))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) >= 2
    finally:
        capture.release()


def test_trigger_does_not_wait_for_slow_mp4_writer(tmp_path):
    cfg = EvidenceConfig(
        enabled=True,
        clips_dir=str(tmp_path),
        pre_seconds=1.0,
        post_seconds=0.0,
        sample_fps=5.0,
        max_buffer_frames=10,
    )
    recorder = EvidenceRecorder(cfg)
    recorder.push("cam", np.full((64, 96, 3), 80, dtype=np.uint8), 20.0)

    writer_started = threading.Event()
    allow_writer = threading.Event()
    trigger_finished = threading.Event()
    original_write_job = recorder._write_job

    def blocked_write(job):
        writer_started.set()
        assert allow_writer.wait(timeout=2.0)
        original_write_job(job)

    recorder._write_job = blocked_write
    result = {}

    def call_trigger():
        result["url"] = recorder.trigger("cam", "nonblocking", 20.0)
        trigger_finished.set()

    caller = threading.Thread(target=call_trigger)
    caller.start()
    try:
        assert trigger_finished.wait(timeout=0.5)
        assert writer_started.wait(timeout=0.5)
        assert result["url"] is not None
    finally:
        allow_writer.set()
        caller.join(timeout=2.0)
        recorder.close()


def test_close_flushes_completed_pre_and_post_clip(tmp_path):
    cfg = EvidenceConfig(
        enabled=True,
        clips_dir=str(tmp_path),
        pre_seconds=1.0,
        post_seconds=0.5,
        sample_fps=4.0,
        max_buffer_frames=10,
    )
    recorder = EvidenceRecorder(cfg)
    recorder.push("cam", np.full((64, 96, 3), 20, dtype=np.uint8), 30.0)
    recorder.push("cam", np.full((64, 96, 3), 60, dtype=np.uint8), 30.25)
    url = recorder.trigger("cam", "with-post", 30.25)
    recorder.push("cam", np.full((64, 96, 3), 100, dtype=np.uint8), 30.5)
    recorder.push("cam", np.full((64, 96, 3), 140, dtype=np.uint8), 30.75)

    assert url is not None
    recorder.close()

    path = tmp_path / url.rsplit("/", 1)[-1]
    assert path.exists()
    assert path.stat().st_size > 0
    capture = cv2.VideoCapture(str(path))
    try:
        # Two pre-event and two post-event samples survive one final encode.
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) >= 4
    finally:
        capture.release()


def test_post_event_trigger_defers_writer_and_encodes_once(tmp_path):
    cfg = EvidenceConfig(
        enabled=True,
        clips_dir=str(tmp_path),
        pre_seconds=1.0,
        post_seconds=0.5,
        sample_fps=4.0,
    )
    recorder = EvidenceRecorder(cfg)
    writes = []
    original = recorder._write_job

    def counted(job):
        writes.append(job)
        original(job)

    recorder._write_job = counted
    recorder.push("cam", np.zeros((64, 96, 3), np.uint8), 10.0)
    recorder.trigger("cam", "single", 10.0)
    try:
        assert len(recorder._write_jobs) == 0
        assert writes == []
        recorder.push("cam", np.ones((64, 96, 3), np.uint8), 10.5)
    finally:
        recorder.close()
    assert len(writes) == 1
