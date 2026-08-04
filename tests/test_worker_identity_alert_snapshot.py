from backend.detector import Detection
from backend.models import AlarmRecord
from backend.routes.ingest import _attach_identity_snapshot
from backend.worker_identification import WorkerIdentity
from backend.worker_store import WorkerRecord


def profile():
    return WorkerRecord("W-001", "Jan", "Kowalski", "Operator", "Budowa", 1.0, 1.0)


def identity(*, status="confirmed", age=0.5, eligible=True):
    person = Detection(0, "person", "person", (1, 2, 30, 60), 0.9, 12)
    return WorkerIdentity(
        "W-001", person, "cache", [], 100.0, True, 12, "marker:1", 0.95,
        marker_id=1, identity_status=status, marker_age_sec=age,
        alert_eligible=eligible,
    )


def record():
    return AlarmRecord(
        id="event", timestamp=100.0, mode="site", kind="fall_detected",
        severity="DANGER", rule_name="fall", description="fall", details={"track_id": 12},
    )


def test_fresh_confirmed_identity_is_frozen_into_event():
    saved = _attach_identity_snapshot(record(), identity(), profile())
    assert saved.details["worker_id"] == "W-001"
    assert saved.details["worker"]["first_name"] == "Jan"
    assert saved.details["marker_id"] == 1
    assert saved.details["identity_status"] == "confirmed"
    original = dict(saved.details)
    changed = identity(status="held", age=4.0, eligible=False)
    assert changed.worker_id == "W-001"
    assert saved.details == original


def test_old_held_or_ambiguous_identity_is_unidentified_for_alert():
    for status in ("held", "ambiguous"):
        saved = _attach_identity_snapshot(
            record(), identity(status=status, age=3.0, eligible=False), profile(),
        )
        assert "worker_id" not in saved.details
        assert saved.details["worker"] is None
        assert saved.details["identity_status"] == "unidentified"
