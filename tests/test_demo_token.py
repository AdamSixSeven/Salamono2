"""Demo token: iframe-friendly auth bypass for pitch presentations."""
import base64
import importlib
import os
import time
from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient, ASGITransport

import backend.main as main_module
from backend.alert_storage import AlertStore
from backend.calibration import CalibrationStore
from backend.danger_rules import DangerDetector, TemporalFilter
from backend.frame_store import FrameStore
from backend.marker_detector import MarkerDetector
from backend.ws_manager import ConnectionManager
from backend.zone_rules import ZoneBreachDetector, ZoneTemporalFilter
from backend.zones_store import ZoneStore
from config import CONFIG


TOKEN = "pitch-demo-secret"
PASSWORD = "shhh"


@pytest.fixture(autouse=True)
def enable_auth_and_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("PANEL_PASSWORD", PASSWORD)
    monkeypatch.setenv("DEMO_TOKEN", TOKEN)
    importlib.reload(main_module)
    app = main_module.app
    det = MagicMock(); det.detect.side_effect = lambda _f: []
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
    yield app
    monkeypatch.undo()
    importlib.reload(main_module)


def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="https://t")


def _basic(pw):
    return "Basic " + base64.b64encode(f":{pw}".encode()).decode()


@pytest.mark.asyncio
async def test_health_public(enable_auth_and_reload):
    async with _client(enable_auth_and_reload) as c:
        r = await c.get("/api/health")
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_missing_auth_returns_401(enable_auth_and_reload):
    async with _client(enable_auth_and_reload) as c:
        r = await c.get("/api/stats")
        assert r.status_code == 401


@pytest.mark.asyncio
async def test_basic_auth_still_works(enable_auth_and_reload):
    async with _client(enable_auth_and_reload) as c:
        r = await c.get("/api/stats", headers={"Authorization": _basic(PASSWORD)})
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_demo_token_query_sets_cookie(enable_auth_and_reload):
    async with _client(enable_auth_and_reload) as c:
        r = await c.get(f"/api/stats?demo={TOKEN}")
        assert r.status_code == 200
        # cookie set with SameSite=None so iframe embeds work
        set_cookie = r.headers.get("set-cookie", "")
        assert "perimetr_demo=" + TOKEN in set_cookie
        assert "samesite=none" in set_cookie.lower()
        assert "secure" in set_cookie.lower()


@pytest.mark.asyncio
async def test_wrong_demo_token_falls_through_to_401(enable_auth_and_reload):
    async with _client(enable_auth_and_reload) as c:
        r = await c.get("/api/stats?demo=wrong")
        assert r.status_code == 401


@pytest.mark.asyncio
async def test_demo_cookie_alone_authorises(enable_auth_and_reload):
    async with _client(enable_auth_and_reload) as c:
        r = await c.get(
            "/api/stats",
            cookies={"perimetr_demo": TOKEN},
        )
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_csp_header_allows_iframe(enable_auth_and_reload):
    async with _client(enable_auth_and_reload) as c:
        r = await c.get("/api/health")
        assert r.headers.get("content-security-policy") == "frame-ancestors *"


@pytest.mark.asyncio
async def test_no_demo_token_env_means_no_bypass(monkeypatch, tmp_path):
    """If DEMO_TOKEN env var is empty (default), ?demo=anything must not
    let a request through."""
    monkeypatch.setenv("PANEL_PASSWORD", PASSWORD)
    monkeypatch.delenv("DEMO_TOKEN", raising=False)
    importlib.reload(main_module)
    app = main_module.app
    det = MagicMock(); det.detect.side_effect = lambda _f: []
    app.state.detector = det
    app.state.danger_detector = DangerDetector()
    app.state.temporal_filter = TemporalFilter(required=1, cooldown_sec=0.1)
    app.state.ws_manager = ConnectionManager()
    app.state.frame_store = FrameStore(str(tmp_path / "flagged"))
    app.state.alert_store = AlertStore(str(tmp_path / "alerts.jsonl"))
    app.state.zone_store = ZoneStore(str(tmp_path / "zones.json"))
    app.state.zone_detector = ZoneBreachDetector()
    app.state.zone_temporal_filter = ZoneTemporalFilter(required=1, cooldown_sec=0.1)
    app.state.marker_detector = MarkerDetector()
    app.state.calibration_store = CalibrationStore(str(tmp_path / "calibration.json"))
    app.state.marker_zone_cache = {}
    app.state.debug_inject_person = None
    app.state.ppe_detector = None
    app.state.ppe_checker = None
    app.state.frame_counter = 0
    app.state.start_time = time.time()

    async with _client(app) as c:
        r = await c.get("/api/stats?demo=anything")
        assert r.status_code == 401

    monkeypatch.undo()
    importlib.reload(main_module)
