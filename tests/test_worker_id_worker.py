import threading

import numpy as np

from backend.detector import Detection
from backend.worker_identification import (
    DecodedWorkerTag,
    WorkerIdentifier,
    WorkerIDWorker,
)
from backend.worker_store import WorkerStore
from config import WorkerIDConfig


def _person(box=(20, 20, 80, 100), track_id=None):
    return Detection(
        class_id=0,
        class_name="person",
        category="person",
        box=box,
        confidence=0.9,
        track_id=track_id,
    )


def _tag(marker_id, polygon=None):
    return DecodedWorkerTag(
        payload=f"marker:{marker_id}",
        polygon=polygon or [(36, 36), (44, 36), (44, 44), (36, 44)],
    )


class SequenceDecoder:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []
        self._lock = threading.Lock()

    def decode(self, frame):
        with self._lock:
            index = len(self.calls)
            self.calls.append(np.array(frame, copy=True))
            if not self.results:
                return []
            return self.results[min(index, len(self.results) - 1)]


class BlockingDecoder:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def decode(self, frame):
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            assert self.release.wait(2.0)
        return [_tag(int(frame[0, 0, 0]) or self.calls)]


class MutableClock:
    def __init__(self, value=0.0):
        self.value = float(value)
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            return self.value

    def advance(self, seconds):
        with self._lock:
            self.value += float(seconds)


def _worker(tmp_path, decoder, **kwargs):
    cfg = WorkerIDConfig(
        enabled=True,
        sample_fps=30.0,
        cache_ttl_seconds=4.0,
        database_path=str(tmp_path / "workers.sqlite3"),
        initial_confirmations=1,
    )
    store = WorkerStore(cfg.database_path)
    for marker_id in range(1, 51):
        store.create(f"W-{marker_id:03d}", "Test", str(marker_id), "", "")
    identifier = WorkerIdentifier(cfg, decoder=decoder)
    return WorkerIDWorker(identifier, cfg, **kwargs)


def test_worker_scans_full_frame_and_maps_live_snapshot_corners(tmp_path):
    decoder = SequenceDecoder([[_tag(7)]])
    worker = _worker(tmp_path, decoder, crop_padding=0.0)
    frame = np.zeros((120, 160, 3), np.uint8)
    try:
        people = worker.submit_latest(
            "cam-a",
            frame,
            [_person((20, 30, 80, 110))],
            10.0,
        )
        snapshot = worker.wait_for_result("cam-a")
        assert snapshot is not None
        assert decoder.calls[0].shape[:2] == (120, 160)
        assert people[0].track_id == 1
        assert snapshot.scanned_track_ids == frozenset({1})
        assert snapshot.identities[0].tag_polygon[0] == (36.0, 36.0)
        assert snapshot.identities[0].track_id == 1

        current = worker.current_identities("cam-a", people, 10.1)
        assert current[0].worker_id == "W-007"
        assert current[0].person is people[0]
        assert current[0].tag_polygon == []
        assert current[0].cached is True
    finally:
        worker.close()


def test_submit_assigns_stable_tracks_and_uses_current_bbox(tmp_path):
    clock = MutableClock()
    worker = _worker(
        tmp_path,
        SequenceDecoder([[_tag(1)]]),
        sample_fps=1.0,
        monotonic_clock=clock,
    )
    frame = np.zeros((160, 200, 3), np.uint8)
    try:
        first = worker.submit_latest("cam", frame, [_person()], 1.0)
        assert worker.wait_for_result("cam") is not None
        moved = [_person((25, 22, 85, 102))]
        second = worker.submit_latest("cam", frame, moved, 1.1)
        assert second[0].track_id == first[0].track_id == 1

        identities = worker.current_identities("cam", second, 1.1)
        assert identities[0].person.box == (25, 22, 85, 102)
        assert identities[0].tag_polygon == []
    finally:
        worker.close()


def test_global_pending_slot_replaces_an_older_camera_job(tmp_path):
    decoder = BlockingDecoder()
    worker = _worker(tmp_path, decoder)
    try:
        first_frame = np.full((120, 120, 3), 1, np.uint8)
        worker.submit_latest("active", first_frame, [_person()], 1.0)
        assert decoder.started.wait(1.0)

        worker.submit_latest(
            "replaced",
            np.full_like(first_frame, 2),
            [_person()],
            2.0,
        )
        worker.submit_latest(
            "newest",
            np.full_like(first_frame, 3),
            [_person()],
            3.0,
        )
        assert worker.pending_count == 1
        decoder.release.set()

        newest = worker.wait_for_result("newest", timeout=2.0)
        assert newest is not None
        assert worker.wait_for_result("replaced", timeout=0.05) is None
        assert decoder.calls == 2
        assert newest.identities[0].worker_id == "W-003"
        assert worker.stats().dropped_jobs >= 1
    finally:
        decoder.release.set()
        worker.close()


def test_reset_camera_rejects_an_in_flight_old_run_result(tmp_path):
    decoder = BlockingDecoder()
    worker = _worker(tmp_path, decoder)
    frame = np.full((120, 120, 3), 8, np.uint8)
    try:
        people = worker.submit_latest("demo", frame, [_person()], 1.0)
        assert decoder.started.wait(1.0)

        worker.reset_camera("demo")
        decoder.release.set()

        assert worker.wait_for_result("demo", timeout=0.2) is None
        assert worker.current_identities("demo", people, 1.1) == []
        assert worker.scanned_track_ids("demo") == set()
    finally:
        decoder.release.set()
        worker.close()


def test_identity_cache_is_partitioned_by_camera_and_track(tmp_path):
    decoder = SequenceDecoder([[_tag(1)], [_tag(2)]])
    worker = _worker(tmp_path, decoder)
    frame = np.zeros((120, 120, 3), np.uint8)
    try:
        people_a = worker.submit_latest("cam-a", frame, [_person()], 1.0)
        assert worker.wait_for_result("cam-a") is not None
        people_b = worker.submit_latest("cam-b", frame, [_person()], 2.0)
        assert worker.wait_for_result("cam-b") is not None

        assert people_a[0].track_id == people_b[0].track_id == 1
        assert worker.current_identities("cam-a", people_a, 2.1)[0].worker_id == "W-001"
        assert worker.current_identities("cam-b", people_b, 2.1)[0].worker_id == "W-002"
    finally:
        worker.close()


def test_cache_expires_using_monotonic_time(tmp_path):
    clock = MutableClock(100.0)
    worker = _worker(
        tmp_path,
        SequenceDecoder([[_tag(4)]]),
        cache_ttl_seconds=4.0,
        monotonic_clock=clock,
    )
    frame = np.zeros((120, 120, 3), np.uint8)
    try:
        people = worker.submit_latest("cam", frame, [_person()], 5000.0)
        assert worker.wait_for_result("cam") is not None
        assert worker.current_identities("cam", people, -1000.0)

        clock.advance(5.01)
        assert worker.current_identities("cam", people, 999999.0) == []
        stats = worker.stats()
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1
    finally:
        worker.close()


def test_no_people_skips_scan_and_removes_disappeared_track(tmp_path):
    decoder = SequenceDecoder([[_tag(5)]])
    worker = _worker(tmp_path, decoder, full_frame_fallback=False)
    frame = np.zeros((120, 120, 3), np.uint8)
    try:
        people = worker.submit_latest("cam", frame, [_person()], 1.0)
        assert worker.wait_for_result("cam") is not None
        assert worker.current_identities("cam", people, 1.1)
        calls_before = len(decoder.calls)

        assert worker.submit_latest("cam", frame, [], 1.2) == []
        assert len(decoder.calls) == calls_before
        assert worker.current_identities("cam", people, 1.3) == []
        assert worker.scanned_track_ids("cam") == set()
    finally:
        worker.close()


def test_multiple_people_are_identified_from_one_full_frame_scan(tmp_path):
    decoder = SequenceDecoder([[
        _tag(11, [(30, 50), (40, 50), (40, 60), (30, 60)]),
        _tag(12, [(170, 60), (180, 60), (180, 70), (170, 70)]),
    ]])
    worker = _worker(tmp_path, decoder, crop_padding=0.0)
    frame = np.zeros((160, 240, 3), np.uint8)
    people = [
        _person((10, 20, 80, 150)),
        _person((140, 30, 220, 150)),
    ]
    try:
        tracked = worker.submit_latest("cam", frame, people, 1.0)
        snapshot = worker.wait_for_result("cam")
        assert snapshot is not None
        assert [person.track_id for person in tracked] == [1, 2]
        assert snapshot.scanned_track_ids == frozenset({1, 2})
        assert len(decoder.calls) == 1
        assert decoder.calls[0].shape[:2] == frame.shape[:2]
        assert worker.stats().crop_count == 1
        assert {
            identity.worker_id
            for identity in worker.current_identities("cam", tracked, 1.1)
        } == {"W-011", "W-012"}
    finally:
        worker.close()


def test_sampling_rate_prevents_repeated_scans(tmp_path):
    clock = MutableClock()
    decoder = SequenceDecoder([[_tag(1)], [_tag(2)]])
    worker = _worker(
        tmp_path,
        decoder,
        sample_fps=1.0,
        monotonic_clock=clock,
    )
    frame = np.zeros((120, 120, 3), np.uint8)
    try:
        worker.submit_latest("cam", frame, [_person()], 1.0)
        assert worker.wait_for_result("cam") is not None
        clock.advance(0.5)
        worker.submit_latest("cam", frame, [_person()], 2.0)
        assert len(decoder.calls) == 1

        clock.advance(0.51)
        worker.submit_latest("cam", frame, [_person()], 3.0)
        assert worker.wait_for_result("cam", after_timestamp=1.0) is not None
        assert len(decoder.calls) == 2
    finally:
        worker.close()


def test_disable_clears_camera_state_and_close_stops_thread(tmp_path):
    worker = _worker(tmp_path, SequenceDecoder([[_tag(9)]]))
    frame = np.zeros((120, 120, 3), np.uint8)
    people = worker.submit_latest("cam", frame, [_person()], 1.0)
    assert worker.wait_for_result("cam") is not None
    assert worker.current_identities("cam", people, 1.1)

    worker.set_camera_enabled("cam", False)
    assert worker.current_identities("cam", people, 1.2) == []
    assert worker.scanned_track_ids("cam") == set()
    worker.close()
    worker.close()
    assert worker.stats().thread_alive is False
