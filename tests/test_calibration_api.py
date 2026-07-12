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
from backend.marker_detector import DEFAULT_DICT, MarkerDetector
from backend.ws_manager import ConnectionManager
from backend.zone_rules import ZoneBreachDetector, ZoneTemporalFilter
from backend.zones_store import ZoneStore
from config import CONFIG


@pytest.fixture(autouse=True)
def init_app_state(tmp_path):
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


def _marker_calibration_image(marker_ids, size_px=180, gap_px=340):
    d = cv2.aruco.getPredefinedDictionary(DEFAULT_DICT)
    positions = [
        (30, 30),                           # TL
        (30 + gap_px, 30),                  # TR
        (30 + gap_px, 30 + gap_px),         # BR
        (30, 30 + gap_px),                  # BL
    ]
    w = 30 + gap_px + size_px + 30
    h = 30 + gap_px + size_px + 30
    frame = np.full((h, w, 3), 255, dtype=np.uint8)
    for mid, (x0, y0) in zip(marker_ids, positions):
        if hasattr(cv2.aruco, "generateImageMarker"):
            img = cv2.aruco.generateImageMarker(d, mid, size_px)
        else:  # pragma: no cover
            img = cv2.aruco.drawMarker(d, mid, size_px)
        frame[y0:y0 + size_px, x0:x0 + size_px] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


def _client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_get_calibration_not_found():
    async with _client() as c:
        resp = await c.get("/api/calibration/cam_a")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_calibrate_and_read_back():
    ids = [1, 2, 3, 4]
    image_bytes = _marker_calibration_image(ids)
    async with _client() as c:
        resp = await c.post(
            "/api/calibration/cam_a",
            files={"image": ("frame.jpg", image_bytes, "image/jpeg")},
            data={
                "marker_ids": "1,2,3,4",
                "width_m": "3.0",
                "height_m": "3.0",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["marker_ids"] == ids
        assert body["width_m"] == 3.0

        got = await c.get("/api/calibration/cam_a")
        assert got.status_code == 200
        assert got.json()["marker_ids"] == ids

        deleted = await c.delete("/api/calibration/cam_a")
        assert deleted.status_code == 200

        # After delete, should be 404 again.
        after = await c.get("/api/calibration/cam_a")
        assert after.status_code == 404


@pytest.mark.asyncio
async def test_calibrate_missing_marker_returns_400():
    image_bytes = _marker_calibration_image([1, 2, 3])  # only 3 markers rendered
    async with _client() as c:
        resp = await c.post(
            "/api/calibration/cam_a",
            files={"image": ("frame.jpg", image_bytes, "image/jpeg")},
            data={
                "marker_ids": "1,2,3,4",
                "width_m": "3.0",
                "height_m": "3.0",
            },
        )
        assert resp.status_code == 400
        assert "not detected" in resp.text.lower()


@pytest.mark.asyncio
async def test_calibrate_wrong_number_of_ids_returns_400():
    image_bytes = _marker_calibration_image([1, 2, 3, 4])
    async with _client() as c:
        resp = await c.post(
            "/api/calibration/cam_a",
            files={"image": ("frame.jpg", image_bytes, "image/jpeg")},
            data={
                "marker_ids": "1,2,3",
                "width_m": "3.0",
                "height_m": "3.0",
            },
        )
        assert resp.status_code == 400
