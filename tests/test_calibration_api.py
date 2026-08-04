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
from backend.latest_frame_store import LatestFrameStore
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
    app.state.latest_frame_store = LatestFrameStore()
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


def test_latest_frame_store_is_bounded_and_replaces_per_camera():
    store = LatestFrameStore(max_cameras=2)
    image = b"valid-enough-for-storage"
    common = {
        "content_type": "image/jpeg",
        "width": 10,
        "height": 10,
        "captured_at": 1.0,
        "received_at": 1.0,
    }
    store.put("cam_a", image + b"-a1", **common)
    store.put("cam_a", image + b"-a2", **common)
    assert len(store) == 1
    assert store.get("cam_a").image_bytes.endswith(b"-a2")

    store.put("cam_b", image + b"-b", **common)
    store.put("cam_c", image + b"-c", **common)
    assert len(store) == 2
    assert store.get("cam_a") is None
    assert store.get("cam_b") is not None
    assert store.get("cam_c") is not None


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
        assert body["source_frame_width"] == 580
        assert body["source_frame_height"] == 580
        assert body["source_aspect_ratio"] == 1.0
        assert len(body["marker_centers"]) == 4
        assert body["quality"]["valid"] is True
        assert body["calibration_quality"] == body["quality"]
        assert body["quality"]["quadrilateral_area_ratio"] > 0.1
        assert body["quality"]["smallest_marker_area_px"] > 100

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


@pytest.mark.asyncio
async def test_latest_camera_frame_is_available_unannotated_for_calibration():
    image_bytes = _marker_calibration_image([1, 2, 3, 4])
    async with _client() as c:
        ingested = await c.post(
            "/api/frame",
            files={"image": ("phone-frame.jpg", image_bytes, "image/jpeg")},
            data={"camera_id": "phone_main"},
        )
        assert ingested.status_code == 200, ingested.text

        latest = await c.get("/api/calibration/phone_main/latest-frame")
        assert latest.status_code == 200
        assert latest.headers["content-type"].startswith("image/jpeg")
        assert latest.headers["cache-control"] == "no-store"
        assert int(latest.headers["x-frame-width"]) > 0
        assert int(latest.headers["x-frame-height"]) > 0
        # The calibration preview returns the original upload: the same
        # unannotated bytes used by live view and calibration, without an
        # extra encode/decode generation.
        assert latest.content == image_bytes


@pytest.mark.asyncio
async def test_latest_camera_frame_can_be_posted_back_for_exact_frame_calibration():
    ids = [11, 12, 13, 14]
    image_bytes = _marker_calibration_image(ids)
    async with _client() as c:
        ingested = await c.post(
            "/api/frame",
            files={"image": ("phone-frame.jpg", image_bytes, "image/jpeg")},
            data={"camera_id": "phone_calibration"},
        )
        assert ingested.status_code == 200, ingested.text

        preview = await c.get("/api/calibration/phone_calibration/latest-frame")
        calibrated = await c.post(
            "/api/calibration/phone_calibration",
            files={"image": ("visible-frame.jpg", preview.content, "image/jpeg")},
            data={
                "marker_ids": "11,12,13,14",
                "width_m": "4.0",
                "height_m": "2.5",
            },
        )
        assert calibrated.status_code == 200, calibrated.text
        assert calibrated.json()["camera_id"] == "phone_calibration"
        assert calibrated.json()["marker_ids"] == ids


@pytest.mark.asyncio
async def test_latest_camera_frame_rejects_missing_or_stale_stream():
    async with _client() as c:
        missing = await c.get("/api/calibration/not_connected/latest-frame")
        assert missing.status_code == 404
        assert "no frame available" in missing.json()["detail"].lower()

        image_bytes = _marker_calibration_image([1, 2, 3, 4])
        app.state.latest_frame_store.put(
            "stale_phone",
            image_bytes,
            content_type="image/jpeg",
            width=550,
            height=550,
            captured_at=100.0,
            received_at=100.0,
        )
        stale = await c.get("/api/calibration/stale_phone/latest-frame")
        assert stale.status_code == 409
        assert "stale" in stale.json()["detail"].lower()
        assert "3.0s" in stale.json()["detail"]


@pytest.mark.asyncio
async def test_calibrate_from_latest_accepts_json_list_and_string_ids():
    ids = [10, 20, 30, 40]
    image_bytes = _marker_calibration_image(ids)
    async with _client() as c:
        ingested = await c.post(
            "/api/frame",
            files={"image": ("phone.jpg", image_bytes, "image/jpeg")},
            data={"camera_id": "phone_json"},
        )
        assert ingested.status_code == 200, ingested.text

        response = await c.post(
            "/api/calibration/phone_json/from-latest",
            json={
                "marker_ids": ids,
                "width_m": 4.0,
                "height_m": 2.0,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["source_aspect_ratio"] == 1.0

        string_response = await c.post(
            "/api/calibration/phone_json/from-latest",
            json={
                "marker_ids": "10,20,30,40",
                "width_m": 4.0,
                "height_m": 2.0,
            },
        )
        assert string_response.status_code == 200, string_response.text


@pytest.mark.asyncio
async def test_calibrate_from_latest_missing_and_stale_are_actionable(monkeypatch):
    async with _client() as c:
        missing = await c.post(
            "/api/calibration/missing/from-latest",
            json={
                "marker_ids": [10, 20, 30, 40],
                "width_m": 3,
                "height_m": 3,
            },
        )
        assert missing.status_code == 404
        assert "no frame available" in missing.json()["detail"].lower()

        image_bytes = _marker_calibration_image([10, 20, 30, 40])
        app.state.latest_frame_store.put(
            "stale_json",
            image_bytes,
            content_type="image/jpeg",
            width=580,
            height=580,
            captured_at=100.0,
            received_at=100.0,
        )
        monkeypatch.setattr("backend.routes.calibration.time.time", lambda: 104.1)
        stale = await c.post(
            "/api/calibration/stale_json/from-latest",
            json={
                "marker_ids": [10, 20, 30, 40],
                "width_m": 3,
                "height_m": 3,
            },
        )
        assert stale.status_code == 409
        assert "4.100s" in stale.json()["detail"]


@pytest.mark.asyncio
async def test_preview_reports_missing_frame_with_200():
    async with _client() as c:
        response = await c.get(
            "/api/calibration/offline/preview",
            params={"marker_ids": "10,20,30,40"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["frame_available"] is False
    assert body["frame_age_sec"] is None
    assert body["detected_markers"] == []
    assert body["missing_marker_ids"] == [10, 20, 30, 40]
    assert body["quality"]["status"] == "invalid"
    assert body["calibration_quality"] == body["quality"]
    assert body["quality"]["messages"]
    assert body["calibration_exists"] is False
    assert body["calibration_valid_for_frame"] is False


@pytest.mark.asyncio
async def test_preview_reports_markers_quality_and_calibration_validity():
    ids = [10, 20, 30, 40]
    image_bytes = _marker_calibration_image(ids)
    async with _client() as c:
        ingested = await c.post(
            "/api/frame",
            files={"image": ("phone.jpg", image_bytes, "image/jpeg")},
            data={"camera_id": "preview_cam"},
        )
        assert ingested.status_code == 200, ingested.text
        calibrated = await c.post(
            "/api/calibration/preview_cam/from-latest",
            json={"marker_ids": ids, "width_m": 3, "height_m": 3},
        )
        assert calibrated.status_code == 200, calibrated.text

        response = await c.get(
            "/api/calibration/preview_cam/preview",
            params={"marker_ids": "10,20,30,40"},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["frame_available"] is True
    assert body["fresh"] is True
    assert body["valid"] is True
    assert body["frame_width"] == 580
    assert body["frame_height"] == 580
    assert {item["marker_id"] for item in body["detected_markers"]} == set(ids)
    assert all(len(item["center"]) == 2 for item in body["detected_markers"])
    assert all(len(item["corners"]) == 4 for item in body["detected_markers"])
    assert body["quality"]["quadrilateral_area_ratio"] > 0.1
    assert body["calibration_exists"] is True
    assert body["calibration_valid_for_frame"] is True


@pytest.mark.asyncio
async def test_preview_stale_frame_returns_metadata_not_http_error(monkeypatch):
    image_bytes = _marker_calibration_image([10, 20, 30, 40])
    app.state.latest_frame_store.put(
        "preview_stale",
        image_bytes,
        content_type="image/jpeg",
        width=580,
        height=580,
        captured_at=100.0,
        received_at=100.0,
    )
    monkeypatch.setattr("backend.routes.calibration.time.time", lambda: 103.5)
    async with _client() as c:
        response = await c.get("/api/calibration/preview_stale/preview")
    assert response.status_code == 200
    body = response.json()
    assert body["frame_available"] is True
    assert body["frame_age_sec"] == pytest.approx(3.5)
    assert body["fresh"] is False
    assert body["valid"] is False
    assert "stale" in body["validation_error"].lower()
    assert body["quality"]["status"] == "invalid"


@pytest.mark.asyncio
async def test_measure_projects_two_points_and_checks_aspect():
    ids = [10, 20, 30, 40]
    image_bytes = _marker_calibration_image(ids)
    async with _client() as c:
        calibrated = await c.post(
            "/api/calibration/measure_cam",
            files={"image": ("frame.jpg", image_bytes, "image/jpeg")},
            data={
                "marker_ids": "10,20,30,40",
                "width_m": "4.0",
                "height_m": "3.0",
            },
        )
        assert calibrated.status_code == 200, calibrated.text
        centers_px = calibrated.json()["marker_centers_px"]

        measured = await c.post(
            "/api/calibration/measure_cam/measure",
            json={
                "point_a": [centers_px[0][0] * 2, centers_px[0][1] * 2],
                "point_b": [centers_px[2][0] * 2, centers_px[2][1] * 2],
                "frame_width": 1160,
                "frame_height": 1160,
            },
        )
        assert measured.status_code == 200, measured.text
        body = measured.json()
        assert body["point_a_m"] == pytest.approx([0.0, 0.0], abs=1e-5)
        assert body["point_b_m"] == pytest.approx([4.0, 3.0], abs=1e-5)
        assert body["distance_m"] == pytest.approx(5.0, abs=1e-5)

        mismatch = await c.post(
            "/api/calibration/measure_cam/measure",
            json={
                "point_a": [120, 120],
                "point_b": [460, 460],
                "frame_width": 640,
                "frame_height": 480,
            },
        )
        assert mismatch.status_code == 409
        assert "aspect mismatch" in mismatch.json()["detail"].lower()


@pytest.mark.asyncio
async def test_marker_png_download_and_range_validation():
    async with _client() as c:
        response = await c.get("/api/calibration/markers/10.png")
        invalid_low = await c.get("/api/calibration/markers/-1.png")
        invalid_high = await c.get("/api/calibration/markers/50.png")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.headers["content-disposition"] == (
        'attachment; filename="aruco_4x4_50_id_10.png"'
    )
    decoded = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert decoded is not None
    assert decoded.shape[0] > 800
    assert decoded.shape[1] > 800
    assert decoded[0, 0] == 255
    assert invalid_low.status_code == 400
    assert invalid_high.status_code == 400
