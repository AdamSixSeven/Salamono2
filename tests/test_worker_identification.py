import numpy as np

from backend.detector import Detection
from backend.worker_identification import (
    DecodedWorkerTag,
    WorkerIdentifier,
)
from config import WorkerIDConfig


class SequenceDecoder:
    def __init__(self, sequences):
        self.sequences = list(sequences)
        self.index = 0

    def decode(self, frame):
        result = self.sequences[min(self.index, len(self.sequences) - 1)]
        self.index += 1
        return result


def person(box=(100, 60, 300, 420)):
    return Detection(0, "person", "person", box, 0.95)


def test_qr_tag_is_matched_to_enclosing_person():
    tag = DecodedWorkerTag(
        "worker:W-001",
        [(170, 150), (230, 150), (230, 210), (170, 210)],
    )
    identifier = WorkerIdentifier(
        WorkerIDConfig(enabled=True, sample_fps=10.0),
        decoder=SequenceDecoder([[tag]]),
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    identities = identifier.process("cam", frame, [person()], 1.0)
    assert len(identities) == 1
    assert identities[0].worker_id == "W-001"
    assert identities[0].cached is False


def test_worker_assignment_survives_short_qr_occlusion():
    tag = DecodedWorkerTag(
        "worker:W-002",
        [(170, 150), (230, 150), (230, 210), (170, 210)],
    )
    identifier = WorkerIdentifier(
        WorkerIDConfig(enabled=True, sample_fps=10.0, cache_ttl_seconds=2.0),
        decoder=SequenceDecoder([[tag], []]),
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    first = identifier.process("cam", frame, [person()], 1.0)
    second = identifier.process("cam", frame, [person((110, 60, 310, 420))], 1.2)
    assert first[0].cached is False
    assert second[0].worker_id == "W-002"
    assert second[0].cached is True


def test_unprefixed_qr_is_ignored():
    tag = DecodedWorkerTag("https://example.com", [(1, 1), (2, 1), (2, 2), (1, 2)])
    identifier = WorkerIdentifier(
        WorkerIDConfig(enabled=True, prefix="worker:"),
        decoder=SequenceDecoder([[tag]]),
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert identifier.process("cam", frame, [person()], 1.0) == []

from backend.worker_identification import UnidentifiedWorkerMonitor, WorkerIdentity


def test_unidentified_worker_requires_consecutive_frames_at_checkpoint():
    monitor = UnidentifiedWorkerMonitor(
        WorkerIDConfig(
            require_at_checkpoint=True,
            require_on_site=False,
            unidentified_frames_required=3,
            unidentified_cooldown_seconds=20.0,
        )
    )
    p = person()
    assert monitor.update("cam", "checkpoint", [p], [], 1.0) == []
    assert monitor.update("cam", "checkpoint", [p], [], 1.2) == []
    events = monitor.update("cam", "checkpoint", [p], [], 1.4)
    assert len(events) == 1
    assert events[0].person == p


def test_identified_worker_never_generates_missing_id_event():
    cfg = WorkerIDConfig(
        require_at_checkpoint=True,
        unidentified_frames_required=1,
    )
    monitor = UnidentifiedWorkerMonitor(cfg)
    p = person()
    identity = WorkerIdentity("W-001", p, "qr", [], 1.0)
    assert monitor.update("cam", "checkpoint", [p], [identity], 1.0) == []


def test_site_unidentified_worker_policy_is_opt_in():
    p = person()
    disabled = UnidentifiedWorkerMonitor(
        WorkerIDConfig(require_on_site=False, unidentified_frames_required=1)
    )
    assert disabled.update("cam", "site", [p], [], 1.0) == []

    enabled = UnidentifiedWorkerMonitor(
        WorkerIDConfig(require_on_site=True, unidentified_frames_required=1)
    )
    assert len(enabled.update("cam", "site", [p], [], 1.0)) == 1


def test_unidentified_worker_respects_cooldown():
    monitor = UnidentifiedWorkerMonitor(
        WorkerIDConfig(
            require_at_checkpoint=True,
            unidentified_frames_required=1,
            unidentified_cooldown_seconds=5.0,
        )
    )
    p = person()
    assert len(monitor.update("cam", "checkpoint", [p], [], 1.0)) == 1
    assert monitor.update("cam", "checkpoint", [p], [], 2.0) == []
    assert len(monitor.update("cam", "checkpoint", [p], [], 6.1)) == 1
