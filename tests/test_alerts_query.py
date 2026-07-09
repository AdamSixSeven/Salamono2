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
