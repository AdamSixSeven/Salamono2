import threading

import numpy as np

from backend.detector import Detection
from backend.worker_identification import DecodedWorkerTag, WorkerIdentifier, WorkerIDWorker
from backend.worker_store import WorkerStore
from config import WorkerIDConfig


class Clock:
    def __init__(self):
        self.value = 0.0
    def __call__(self):
        return self.value


class Decoder:
    def __init__(self, results):
        self.results = results
        self.calls = []
        self.lock = threading.Lock()
    def decode(self, frame):
        with self.lock:
            self.calls.append(frame.shape[:2])
            return self.results[min(len(self.calls) - 1, len(self.results) - 1)]


def person(box, track_id):
    return Detection(0, "person", "person", box, 0.9, track_id)


def tag(marker_id, x, y):
    return DecodedWorkerTag(
        f"marker:{marker_id}",
        [(x - 4, y - 4), (x + 4, y - 4), (x + 4, y + 4), (x - 4, y + 4)],
    )


def make_worker(tmp_path, results, worker_ids=("W-001", "W-002")):
    cfg = WorkerIDConfig(database_path=str(tmp_path / "workers.sqlite3"), sample_fps=10)
    store = WorkerStore(cfg.database_path)
    for worker_id in worker_ids:
        store.create(worker_id, "Jan", worker_id, "", "")
    clock = Clock()
    decoder = Decoder(results)
    worker = WorkerIDWorker(WorkerIdentifier(cfg, decoder), cfg, monotonic_clock=clock)
    return worker, decoder, clock


def scan(worker, clock, frame, people, timestamp):
    tracked = worker.submit_latest("cam", frame, people, timestamp)
    result = worker.wait_for_result("cam", after_timestamp=timestamp - 0.1, timeout=2)
    assert result is not None
    clock.value += 1.0
    return tracked, result


def test_full_frame_scan_identifies_two_workers_after_two_confirmations(tmp_path):
    detections = [tag(1, 50, 60), tag(2, 170, 60)]
    worker, decoder, clock = make_worker(tmp_path, [detections, detections])
    frame = np.zeros((120, 240, 3), np.uint8)
    people = [person((10, 10, 100, 115), 10), person((130, 10, 230, 115), 20)]
    try:
        scan(worker, clock, frame, people, 1.0)
        assert worker.current_identities("cam", people, 1.1) == []
        tracked, _ = scan(worker, clock, frame, people, 2.0)
        assert {i.worker_id for i in worker.current_identities("cam", tracked, 2.1)} == {"W-001", "W-002"}
        assert decoder.calls == [(120, 240), (120, 240)]
    finally:
        worker.close()


def test_unknown_marker_is_ignored_without_synthetic_worker_id(tmp_path):
    worker, _, clock = make_worker(tmp_path, [[tag(77, 50, 60)]], worker_ids=("W-001",))
    frame = np.zeros((120, 120, 3), np.uint8)
    people = [person((10, 10, 110, 115), 10)]
    try:
        tracked, result = scan(worker, clock, frame, people, 1.0)
        assert result.identities == ()
        assert worker.current_identities("cam", tracked, 1.1) == []
    finally:
        worker.close()


def test_overlapping_boxes_with_similar_score_fail_closed(tmp_path):
    marker = tag(1, 60, 60)
    worker, _, clock = make_worker(tmp_path, [[marker], [marker]], worker_ids=("W-001",))
    frame = np.zeros((120, 120, 3), np.uint8)
    people = [person((10, 10, 105, 115), 10), person((15, 10, 110, 115), 20)]
    try:
        scan(worker, clock, frame, people, 1.0)
        tracked, result = scan(worker, clock, frame, people, 2.0)
        assert result.identities == ()
        assert worker.current_identities("cam", tracked, 2.1) == []
    finally:
        worker.close()


def test_candidate_confirmed_held_and_separate_alert_ttl(tmp_path):
    marker = tag(1, 50, 60)
    worker, _, clock = make_worker(tmp_path, [[marker], [marker], []], worker_ids=("W-001",))
    frame = np.zeros((120, 120, 3), np.uint8)
    people = [person((10, 10, 110, 115), 10)]
    try:
        scan(worker, clock, frame, people, 1.0)
        assert worker.identity_state("cam", 10).status == "candidate"
        tracked, _ = scan(worker, clock, frame, people, 2.0)
        assert worker.identity_state("cam", 10).status == "confirmed"
        tracked, _ = scan(worker, clock, frame, people, 3.0)
        shown = worker.current_identities("cam", tracked, 3.1)[0]
        assert shown.identity_status == "held"
        assert shown.worker_id == "W-001"
        clock.value += 2.1
        shown = worker.current_identities("cam", tracked, 5.2)[0]
        assert shown.identity_status == "held"
        assert shown.alert_eligible is False
        clock.value += 3.0
        assert worker.current_identities("cam", tracked, 8.2) == []
    finally:
        worker.close()


def test_single_wrong_marker_does_not_switch_but_four_do(tmp_path):
    old = tag(1, 50, 60)
    new = tag(2, 50, 60)
    worker, _, clock = make_worker(tmp_path, [[old], [old], [new], [old], [new], [new], [new], [new]])
    frame = np.zeros((120, 120, 3), np.uint8)
    people = [person((10, 10, 110, 115), 10)]
    try:
        for index in range(2):
            scan(worker, clock, frame, people, float(index + 1))
        scan(worker, clock, frame, people, 3.0)
        assert worker.identity_state("cam", 10).status == "switch_pending"
        assert worker.identity_state("cam", 10).worker_id == "W-001"
        scan(worker, clock, frame, people, 4.0)
        assert worker.identity_state("cam", 10).status == "confirmed"
        assert worker.identity_state("cam", 10).worker_id == "W-001"
        for timestamp in (5.0, 6.0, 7.0, 8.0):
            scan(worker, clock, frame, people, timestamp)
        assert worker.identity_state("cam", 10).worker_id == "W-002"
        assert worker.identity_state("cam", 10).status == "confirmed"
    finally:
        worker.close()


def test_unique_worker_moves_from_stale_hold_only_after_new_confirmation(tmp_path):
    marker = tag(1, 50, 60)
    worker, _, clock = make_worker(tmp_path, [[marker], [marker], [marker], [marker]], worker_ids=("W-001",))
    frame = np.zeros((120, 220, 3), np.uint8)
    first = [person((10, 10, 100, 115), 10)]
    second = [person((10, 10, 100, 115), 20)]
    try:
        scan(worker, clock, frame, first, 1.0)
        scan(worker, clock, frame, first, 2.0)
        # Force a distinct track; the gap exceeds conservative reacquire.
        clock.value += 1.0
        scan(worker, clock, frame, second, 3.0)
        assert worker.identity_state("cam", 20).status == "candidate"
        scan(worker, clock, frame, second, 4.0)
        assert worker.identity_state("cam", 20).status == "confirmed"
        old = worker.identity_state("cam", 10)
        assert old is None or old.status == "expired"
    finally:
        worker.close()


def test_overlap_invalidates_display_until_marker_is_reconfirmed(tmp_path):
    detections = [tag(1, 45, 55), tag(2, 165, 55)]
    worker, _, clock = make_worker(tmp_path, [detections, detections, detections, detections])
    frame = np.zeros((120, 220, 3), np.uint8)
    separate = [person((5, 5, 95, 115), 10), person((125, 5, 215, 115), 20)]
    overlap = [person((30, 5, 150, 115), 10), person((55, 5, 175, 115), 20)]
    try:
        scan(worker, clock, frame, separate, 1.0)
        tracked, _ = scan(worker, clock, frame, separate, 2.0)
        assert len(worker.current_identities("cam", tracked, 2.1)) == 2
        worker.submit_latest("cam", frame, overlap, 2.2)
        assert worker.current_identities("cam", overlap, 2.2) == []
        scan(worker, clock, frame, separate, 3.0)
        assert worker.current_identities("cam", separate, 3.1) == []
        tracked, _ = scan(worker, clock, frame, separate, 4.0)
        assert len(worker.current_identities("cam", tracked, 4.1)) == 2
    finally:
        worker.close()


def test_camera_reset_removes_identity_state(tmp_path):
    marker = tag(1, 50, 60)
    worker, _, clock = make_worker(tmp_path, [[marker], [marker]], worker_ids=("W-001",))
    frame = np.zeros((120, 120, 3), np.uint8)
    people = [person((10, 10, 110, 115), 10)]
    try:
        scan(worker, clock, frame, people, 1.0)
        scan(worker, clock, frame, people, 2.0)
        worker.reset_camera("cam")
        assert worker.identity_state("cam", 10) is None
    finally:
        worker.close()


def test_single_person_new_track_is_reacquired_only_as_held(tmp_path):
    marker = tag(1, 50, 60)
    worker, _, clock = make_worker(tmp_path, [[marker], [marker]], worker_ids=("W-001",))
    frame = np.zeros((120, 120, 3), np.uint8)
    old = [person((10, 10, 110, 115), 10)]
    try:
        scan(worker, clock, frame, old, 1.0)
        scan(worker, clock, frame, old, 2.0)
        clock.value = worker.identity_state("cam", 10).last_track_seen_at + 0.1
        new = worker.submit_latest("cam", frame, [person((12, 10, 112, 115), 99)], 2.1)
        shown = worker.current_identities("cam", new, 2.1)
        assert len(shown) == 1
        assert shown[0].track_id == 99
        assert shown[0].identity_status == "held"
        assert worker.identity_state("cam", 10) is None
    finally:
        worker.close()


def test_two_people_disable_reacquire(tmp_path):
    marker = tag(1, 50, 60)
    worker, _, clock = make_worker(tmp_path, [[marker], [marker]], worker_ids=("W-001",))
    frame = np.zeros((120, 240, 3), np.uint8)
    old = [person((10, 10, 110, 115), 10)]
    try:
        scan(worker, clock, frame, old, 1.0)
        scan(worker, clock, frame, old, 2.0)
        clock.value = worker.identity_state("cam", 10).last_track_seen_at + 0.1
        people = [person((12, 10, 112, 115), 99), person((130, 10, 230, 115), 100)]
        tracked = worker.submit_latest("cam", frame, people, 2.1)
        assert worker.current_identities("cam", tracked, 2.1) == []
        assert worker.identity_state("cam", 99) is None
    finally:
        worker.close()


def test_deleted_worker_profile_invalidates_active_display(tmp_path):
    marker = tag(1, 50, 60)
    worker, _, clock = make_worker(tmp_path, [[marker], [marker]], worker_ids=("W-001",))
    frame = np.zeros((120, 120, 3), np.uint8)
    people = [person((10, 10, 110, 115), 10)]
    try:
        scan(worker, clock, frame, people, 1.0)
        tracked, _ = scan(worker, clock, frame, people, 2.0)
        worker.identifier._worker_store.delete("W-001")
        assert worker.current_identities("cam", tracked, 2.1) == []
        assert worker.identity_state("cam", 10) is None
    finally:
        worker.close()
