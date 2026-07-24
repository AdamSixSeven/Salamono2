import os
import sys
import time

import pytest
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.alert_storage import AlertStore
from backend.calibration import CalibrationStore
from backend.danger_rules import DangerDetector, TemporalFilter
from backend.frame_store import FrameStore
from backend.main import app
from backend.marker_detector import MarkerDetector
from backend.models import AlarmRecord, AlertSeverity
from backend.ws_manager import ConnectionManager
from backend.zone_rules import ZoneBreachDetector, ZoneTemporalFilter
from backend.zones_store import ZoneStore
from config import CONFIG


def _make_record(rid: str, ts: float, kind: str, severity: str,
                 mode: str = "site") -> AlarmRecord:
    return AlarmRecord(
        id=rid,
        timestamp=ts,
        mode=mode,
        kind=kind,
        severity=AlertSeverity(severity),
        rule_name=f"{kind}_rule",
        description=f"Test {kind} {severity}",
        camera_id="cam_default",
    )


@pytest.fixture(autouse=True)
def init_app_state(tmp_path):
    from unittest.mock import MagicMock
    det = MagicMock()
    det.detect.return_value = []
    app.state.detector = det
    app.state.danger_detector = DangerDetector()
    app.state.temporal_filter = TemporalFilter(
        required=CONFIG.danger.consecutive_frames_required,
        cooldown_sec=CONFIG.danger.cooldown_seconds,
    )
    app.state.ws_manager = ConnectionManager()
    app.state.frame_store = FrameStore(str(tmp_path / "flagged"))
    app.state.alert_store = AlertStore(str(tmp_path / "alerts.jsonl"))
    app.state.zone_store = ZoneStore(str(tmp_path / "zones.json"))
    app.state.zone_detector = ZoneBreachDetector()
    app.state.zone_temporal_filter = ZoneTemporalFilter(required=1, cooldown_sec=0.5)
    app.state.marker_detector = MarkerDetector()
    app.state.calibration_store = CalibrationStore(str(tmp_path / "calibration.json"))
    app.state.marker_zone_cache = {}
    app.state.debug_inject_person = None
    app.state.ppe_detector = None
    app.state.ppe_checker = None
    app.state.frame_counter = 0
    app.state.start_time = time.time()


def _client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_alerts_filter_by_kind():
    store = app.state.alert_store
    store.append(_make_record("a1", 1000.0, "ppe_missing", "DANGER", mode="checkpoint"))
    store.append(_make_record("a2", 1001.0, "site_hazard", "WARNING"))
    store.append(_make_record("a3", 1002.0, "zone_breach", "DANGER"))
    store.append(_make_record("a4", 1003.0, "ppe_missing", "DANGER", mode="checkpoint"))

    async with _client() as c:
        resp = await c.get("/api/alerts?kind=ppe_missing")
        assert resp.status_code == 200
        rows = resp.json()
        assert len(rows) == 2
        assert {r["id"] for r in rows} == {"a1", "a4"}


@pytest.mark.asyncio
async def test_alerts_summary_counts():
    store = app.state.alert_store
    store.append(_make_record("a1", 1000.0, "ppe_missing", "DANGER", mode="checkpoint"))
    store.append(_make_record("a2", 1001.0, "site_hazard", "WARNING"))
    store.append(_make_record("a3", 1002.0, "zone_breach", "DANGER"))
    store.append(_make_record("a4", 1003.0, "ppe_missing", "DANGER", mode="checkpoint"))

    async with _client() as c:
        resp = await c.get("/api/alerts/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 4
        assert body["by_severity"]["DANGER"] == 3
        assert body["by_severity"]["WARNING"] == 1
        assert body["by_kind"]["ppe_missing"] == 2
        assert body["by_kind"]["site_hazard"] == 1
        assert body["by_kind"]["zone_breach"] == 1


@pytest.mark.asyncio
async def test_alerts_summary_since_until():
    store = app.state.alert_store
    store.append(_make_record("old", 500.0, "site_hazard", "WARNING"))
    store.append(_make_record("new", 1500.0, "site_hazard", "DANGER"))

    async with _client() as c:
        resp = await c.get("/api/alerts/summary?since=1000")
        body = resp.json()
        assert body["total"] == 1
        assert body["by_severity"]["DANGER"] == 1
        assert "WARNING" not in body["by_severity"]

@pytest.mark.asyncio
async def test_alerts_filter_by_worker_and_summary():
    store = app.state.alert_store
    rec_a = _make_record("w1", 1700.0, "site_hazard", "DANGER")
    rec_a.details = {"worker_id": "W-001", "distance_m": 1.2}
    rec_b = _make_record("w2", 1701.0, "fall_detected", "DANGER")
    rec_b.details = {"worker_id": "W-002"}
    rec_c = _make_record("w3", 1702.0, "ppe_missing", "DANGER", mode="checkpoint")
    rec_c.details = {"worker_id": "W-001"}
    store.extend([rec_a, rec_b, rec_c])

    async with _client() as c:
        resp = await c.get("/api/alerts?worker_id=W-001")
        assert resp.status_code == 200
        assert {row["id"] for row in resp.json()} == {"w1", "w3"}

        summary = await c.get("/api/workers/summary")
        assert summary.status_code == 200
        by_id = {row["worker_id"]: row for row in summary.json()["workers"]}
        assert by_id["W-001"]["events"] == 2
        assert by_id["W-001"]["danger_events"] == 2


@pytest.mark.asyncio
async def test_report_csv_contains_worker_and_details():
    rec = _make_record("csv1", 1800.0, "site_hazard", "DANGER")
    rec.details = {
        "worker_id": "W-CSV",
        "distance_m": 1.25,
        "calibrated": True,
    }
    app.state.alert_store.append(rec)

    async with _client() as c:
        resp = await c.get("/api/reports/export.csv?worker_id=W-CSV")
    assert resp.status_code == 200
    assert resp.content.startswith(b"\xef\xbb\xbf")
    text = resp.content.decode("utf-8-sig")
    assert "worker_id" in text
    assert "W-CSV" in text
    assert '""distance_m"":1.25' in text
    assert "attachment;" in resp.headers["content-disposition"]


@pytest.mark.asyncio
async def test_worker_qr_endpoint_returns_svg():
    async with _client() as c:
        resp = await c.get("/api/worker-qr?worker_id=W-001")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/svg+xml")
    assert b"<svg" in resp.content


@pytest.mark.asyncio
async def test_alert_review_workflow_and_filter():
    app.state.alert_store.append(_make_record("review1", 2000.0, "zone_breach", "DANGER"))
    async with _client() as c:
        resp = await c.patch("/api/alerts/review1/review", json={
            "status": "confirmed",
            "reviewed_by": "Kierownik BHP",
            "note": "Potwierdzone na nagraniu.",
        })
        assert resp.status_code == 200
        updated = resp.json()
        assert updated["review_status"] == "confirmed"
        assert updated["reviewed_by"] == "Kierownik BHP"
        assert updated["reviewed_at"] is not None

        filtered = await c.get("/api/alerts?review_status=confirmed")
        assert [row["id"] for row in filtered.json()] == ["review1"]
        summary = (await c.get("/api/alerts/summary")).json()
        assert summary["by_review_status"]["confirmed"] == 1


@pytest.mark.asyncio
async def test_training_export_uses_human_review_labels():
    confirmed = _make_record("train-ok", 2100.0, "posture_anomaly", "WARNING")
    false_alarm = _make_record("train-no", 2101.0, "posture_anomaly", "WARNING")
    app.state.alert_store.extend([confirmed, false_alarm])
    app.state.alert_store.update_review("train-ok", "confirmed", note="chwiejny chód")
    app.state.alert_store.update_review("train-no", "false_positive", note="niósł ciężar")

    async with _client() as c:
        response = await c.get("/api/reports/training.jsonl?review_status=confirmed")
    assert response.status_code == 200
    text = response.text
    assert "train-ok" in text
    assert "train-no" not in text
    assert '"label":"confirmed"' in text
