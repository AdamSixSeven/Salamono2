"""Marker-defined zones: polygon resolved from live ArUco detections."""
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
from backend.detector import Detection
from backend.frame_store import FrameStore
from backend.main import app
from backend.marker_detector import DEFAULT_DICT, MarkerDetection, MarkerDetector
from backend.routes.ingest import _resolve_marker_zones
from backend.ws_manager import ConnectionManager
from backend.zone_rules import ZoneBreachDetector, ZoneTemporalFilter
from backend.zones_store import Zone, ZoneStore
from config import CONFIG


def _mk_marker(mid: int, cx: float, cy: float) -> MarkerDetection:
    # Corners collapsed to the same point — resolver only uses the centre.
    pt = (cx, cy)
    return MarkerDetection(marker_id=mid, corners=[pt, pt, pt, pt])


def test_regular_zone_pass_through():
    z = Zone(id="z1", name="Test", polygon=[[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]])
    out = _resolve_marker_zones([z], [], "cam", {}, now=1.0, frame_w=100, frame_h=100)
    assert len(out) == 1
    assert out[0].polygon == [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]]


def test_marker_zone_all_markers_visible():
    z = Zone(id="mz1", name="Wykop", marker_ids=[10, 20, 30, 40])
    markers = [
        _mk_marker(10, 10, 10),   # TL
        _mk_marker(20, 90, 10),   # TR
        _mk_marker(30, 90, 90),   # BR
        _mk_marker(40, 10, 90),   # BL
    ]
    cache = {}
    out = _resolve_marker_zones([z], markers, "cam", cache,
                                now=1.0, frame_w=100, frame_h=100)
    assert len(out) == 1
    poly = out[0].polygon
    assert poly == [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
    # Cache was populated for fallback next frame.
    assert ("cam", "mz1") in cache


def test_marker_zone_falls_back_to_cache_within_ttl():
    z = Zone(id="mz1", name="Wykop", marker_ids=[10, 20, 30, 40])
    cache = {("cam", "mz1"): ([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]], 100.0)}
    # No markers visible now, but cache is 1 s old (< 2 s TTL) — reuse.
    out = _resolve_marker_zones([z], [], "cam", cache,
                                now=101.0, frame_w=100, frame_h=100)
    assert len(out) == 1
    assert out[0].polygon == [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


def test_marker_zone_dropped_when_cache_stale():
    z = Zone(id="mz1", name="Wykop", marker_ids=[10, 20, 30, 40])
    cache = {("cam", "mz1"): ([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], 100.0)}
    # 5 s later, no markers visible, cache older than TTL — drop zone.
    out = _resolve_marker_zones([z], [], "cam", cache,
                                now=105.0, frame_w=100, frame_h=100)
    assert out == []


def test_marker_zone_dropped_when_no_cache_and_partial_visibility():
    z = Zone(id="mz1", name="Wykop", marker_ids=[10, 20, 30, 40])
    # Only 3 of 4 markers visible → not enough, no cache → drop.
    markers = [_mk_marker(10, 10, 10), _mk_marker(20, 90, 10), _mk_marker(30, 90, 90)]
    out = _resolve_marker_zones([z], markers, "cam", {},
                                now=1.0, frame_w=100, frame_h=100)
    assert out == []


def test_multiple_zones_independent_caches():
    z_a = Zone(id="a", name="A", marker_ids=[1, 2, 3])
    z_b = Zone(id="b", name="B", marker_ids=[4, 5, 6])
    # Only zone A's markers visible.
    markers = [_mk_marker(1, 10, 10), _mk_marker(2, 90, 10), _mk_marker(3, 50, 90)]
    cache = {}
    out = _resolve_marker_zones([z_a, z_b], markers, "cam", cache,
                                now=1.0, frame_w=100, frame_h=100)
    assert [z.id for z in out] == ["a"]
    assert ("cam", "a") in cache
    assert ("cam", "b") not in cache


def _marker_frame_jpeg(marker_ids, size_px=200, gap_px=400):
    """4 markers in the corners of a big BGR frame."""
    d = cv2.aruco.getPredefinedDictionary(DEFAULT_DICT)
    positions = [
        (40, 40),
        (40 + gap_px, 40),
        (40 + gap_px, 40 + gap_px),
        (40, 40 + gap_px),
    ]
    w = 40 + gap_px + size_px + 40
    h = 40 + gap_px + size_px + 40
    frame = np.full((h, w, 3), 255, dtype=np.uint8)
    for mid, (x0, y0) in zip(marker_ids, positions):
        if hasattr(cv2.aruco, "generateImageMarker"):
            img = cv2.aruco.generateImageMarker(d, mid, size_px)
        else:  # pragma: no cover
            img = cv2.aruco.drawMarker(d, mid, size_px)
        frame[y0:y0 + size_px, x0:x0 + size_px] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


@pytest.fixture
def frame_app_state(tmp_path):
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


@pytest.mark.asyncio
async def test_frame_response_contains_active_zone_polygon(frame_app_state):
    """When a marker-zone is configured and the frame shows all 4 markers,
    the /api/frame response must include the resolved polygon so the
    frontend can draw it live."""
    ids = [10, 20, 30, 40]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Register a marker-defined zone.
        r = await c.put("/api/zones/cam_default", json={
            "zones": [{
                "name": "Wykop A",
                "severity": "DANGER",
                "polygon": [],
                "marker_ids": ids,
                "active": True,
            }],
        })
        assert r.status_code == 200, r.text
        zone_id = r.json()["zones"][0]["id"]

        # Send a frame with all 4 markers visible.
        jpeg = _marker_frame_jpeg(ids)
        r = await c.post(
            "/api/frame",
            files={"image": ("frame.jpg", jpeg, "image/jpeg")},
            data={"camera_id": "cam_default", "timestamp": "1000.0"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "site"
        assert "active_zones" in body
        active = body["active_zones"]
        # Should contain our zone with a resolved polygon of 4 vertices.
        by_id = {z["id"]: z for z in active}
        assert zone_id in by_id
        z = by_id[zone_id]
        assert len(z["polygon"]) == 4
        assert z["marker_ids"] == ids
        # All polygon coords should be in [0, 1].
        for x, y in z["polygon"]:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0
