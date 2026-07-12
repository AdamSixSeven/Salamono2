"""Debug endpoint: inject a synthetic person for N frames."""
import os
import sys
import time
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.alert_storage import AlertStore
from backend.calibration import CalibrationStore
from backend.danger_rules import DangerDetector, TemporalFilter
from backend.frame_store import FrameStore
from backend.main import app
from backend.marker_detector import MarkerDetector
from backend.ws_manager import ConnectionManager
from backend.zone_rules import ZoneBreachDetector, ZoneTemporalFilter
from backend.zones_store import Zone, ZoneStore
from config import CONFIG


@pytest.fixture(autouse=True)
def init_app_state(tmp_path):
    det = MagicMock()
    # side_effect returns a fresh list per call; return_value would be the
    # same list mutated by _handle_site's `detections.append(fake)`.
    det.detect.side_effect = lambda _f: []
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


def _blank_jpeg(w=640, h=640):
    frame = np.full((h, w, 3), 255, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


def _client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


@pytest.mark.asyncio
async def test_inject_status_idle_by_default():
    async with _client() as c:
        r = await c.get("/api/debug/inject-person")
        assert r.status_code == 200
        assert r.json()["status"] == "idle"


@pytest.mark.asyncio
async def test_inject_validates_box():
    async with _client() as c:
        r = await c.post("/api/debug/inject-person", json={
            "box_norm": [0.5, 0.5, 0.3, 0.7],   # x1 > x2 invalid
            "frames": 5,
        })
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_inject_produces_person_in_detections():
    async with _client() as c:
        r = await c.post("/api/debug/inject-person", json={
            "box_norm": [0.4, 0.4, 0.6, 0.8],
            "frames": 1,
        })
        assert r.status_code == 200
        assert r.json()["status"] == "armed"

        jpeg = _blank_jpeg()
        r1 = await c.post(
            "/api/frame",
            files={"image": ("f.jpg", jpeg, "image/jpeg")},
            data={"camera_id": "cam_default", "timestamp": "1000"},
        )
        assert r1.status_code == 200
        persons = [d for d in r1.json()["detections"] if d["category"] == "person"]
        assert len(persons) == 1

        # After the single frame budget is spent, injection stops.
        status = await c.get("/api/debug/inject-person")
        assert status.json()["status"] == "idle"

        r2 = await c.post(
            "/api/frame",
            files={"image": ("f.jpg", jpeg, "image/jpeg")},
            data={"camera_id": "cam_default", "timestamp": "1001"},
        )
        persons2 = [d for d in r2.json()["detections"] if d["category"] == "person"]
        assert len(persons2) == 0


@pytest.mark.asyncio
async def test_inject_triggers_zone_breach():
    """End-to-end: injected person standing inside a saved polygon zone
    must produce a confirmed zone breach and an alert record."""
    zone_store = app.state.zone_store
    zone_store.replace("cam_default", [Zone(
        id="zone_test",
        name="Test",
        severity="DANGER",
        polygon=[[0.3, 0.3], [0.7, 0.3], [0.7, 0.9], [0.3, 0.9]],
        active=True,
    )])

    async with _client() as c:
        await c.post("/api/debug/inject-person", json={
            "box_norm": [0.4, 0.5, 0.6, 0.85],
            "frames": 5,
        })

        jpeg = _blank_jpeg()
        breach_seen = False
        for i in range(5):
            r = await c.post(
                "/api/frame",
                files={"image": ("f.jpg", jpeg, "image/jpeg")},
                data={"camera_id": "cam_default", "timestamp": str(1000 + i)},
            )
            body = r.json()
            if body.get("confirmed_zone_breaches"):
                breach_seen = True
                assert body["confirmed_zone_breaches"][0]["zone_name"] == "Test"
                break

        assert breach_seen, "confirmed_zone_breach should have fired"


@pytest.mark.asyncio
async def test_inject_can_be_cleared():
    async with _client() as c:
        await c.post("/api/debug/inject-person", json={
            "box_norm": [0.4, 0.4, 0.6, 0.8],
            "frames": 100,
        })
        r = await c.delete("/api/debug/inject-person")
        assert r.status_code == 200
        status = await c.get("/api/debug/inject-person")
        assert status.json()["status"] == "idle"
