import base64
import sys
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import cv2
import pytest
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app
from backend.alert_storage import AlertStore
from backend.calibration import CalibrationStore
from backend.detector import Detection
from backend.danger_rules import DangerDetector, TemporalFilter
from backend.marker_detector import MarkerDetector
from backend.ws_manager import ConnectionManager
from backend.frame_store import FrameStore
from backend.latest_frame_store import LatestFrameStore
from backend.zone_rules import ZoneBreachDetector, ZoneTemporalFilter
from backend.zones_store import ZoneStore
from config import CONFIG


def _mock_detector():
    det = MagicMock()
    det.detect.return_value = [
        Detection(class_id=0, class_name="person", category="person",
                  box=(100, 100, 200, 300), confidence=0.85),
        Detection(class_id=7, class_name="truck", category="vehicle",
                  box=(400, 100, 600, 300), confidence=0.92),
    ]
    return det


@pytest.fixture(autouse=True)
def init_app_state(tmp_path):
    app.state.detector = _mock_detector()
    app.state.danger_detector = DangerDetector()
    app.state.temporal_filter = TemporalFilter(
        required=CONFIG.danger.consecutive_frames_required,
        cooldown_sec=CONFIG.danger.cooldown_seconds,
    )
    app.state.ws_manager = ConnectionManager()
    app.state.frame_store = FrameStore(str(tmp_path / "flagged"))
    app.state.latest_frame_store = LatestFrameStore(max_cameras=16)
    app.state.alert_store = AlertStore(str(tmp_path / "alerts.jsonl"))
    app.state.zone_store = ZoneStore(str(tmp_path / "zones.json"))
    app.state.zone_detector = ZoneBreachDetector()
    app.state.zone_temporal_filter = ZoneTemporalFilter(
        required=CONFIG.danger.consecutive_frames_required,
        cooldown_sec=CONFIG.danger.cooldown_seconds,
    )
    app.state.marker_detector = MarkerDetector()
    app.state.calibration_store = CalibrationStore(str(tmp_path / "calibration.json"))
    app.state.marker_zone_cache = {}
    app.state.debug_inject_person = None
    app.state.ppe_detector = None
    app.state.ppe_checker = None
    app.state.frame_counter = 0
    app.state.camera_registry = {}
    app.state.start_time = time.time()


def _make_test_jpeg(width=640, height=480):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 300), (0, 255, 0), -1)
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_frontend_assets_revalidate_and_use_matching_versions():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        page = await client.get("/")
        stylesheet = await client.get("/style.css?v=20260724-live-layers-6")

    assert page.status_code == 200
    assert stylesheet.status_code == 200
    assert 'style.css?v=20260724-live-layers-6' in page.text
    assert page.headers["cache-control"] == "no-cache, must-revalidate"
    assert stylesheet.headers["cache-control"] == "no-cache, must-revalidate"


@pytest.mark.asyncio
async def test_frontend_layers_have_independent_client_renderers():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        page = await client.get("/")
        app_asset = await client.get("/app.js?v=20260724-live-layers-6")
        overlay_asset = await client.get("/zones.js?v=20260724-live-layers-6")

    assert page.status_code == 200
    assert app_asset.status_code == 200
    assert overlay_asset.status_code == 200
    assert 'data-layer="posture"' in page.text
    assert ">POS</span>" in page.text
    assert "window.Perimetr.getLayers" in app_asset.text
    assert "posture: parsed.posture !== false" in app_asset.text
    assert 'new CustomEvent("perimetr-frame-rendered"' in app_asset.text
    assert "if (!layers.boxes) return;" in overlay_asset.text
    assert "if (!layers.posture) return;" in overlay_asset.text
    assert "if (!layers.distances) return;" in overlay_asset.text
    assert "if (layers.zones)" in overlay_asset.text
    assert "if (!layers.markers) return;" in overlay_asset.text
    assert "return hasLiveZoneState ? liveActiveZones : zones;" in overlay_asset.text


@pytest.mark.asyncio
async def test_stats():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_frames_processed" in data
        assert "uptime_seconds" in data
        assert data["total_frames_processed"] == 0


@pytest.mark.asyncio
async def test_post_frame():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        jpeg = _make_test_jpeg()
        resp = await client.post(
            "/api/frame",
            files={"image": ("frame.jpg", jpeg, "image/jpeg")},
            data={"camera_id": "test", "timestamp": "1000.0"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["frame_id"] == 1
        assert len(data["detections"]) == 2
        assert data["frame_jpeg_b64"] != ""
        # The live-view image is the untouched upload.  Detection metadata is
        # sent separately so BBOX can be hidden locally without another API
        # request and without a rectangle remaining baked into the pixels.
        assert base64.b64decode(data["frame_jpeg_b64"]) == jpeg
        assert data["processing_ms"] > 0
        categories = {d["category"] for d in data["detections"]}
        assert "person" in categories
        assert "vehicle" in categories


@pytest.mark.asyncio
async def test_frame_rejects_decoded_image_larger_than_calibration_store_limit():
    jpeg = _make_test_jpeg(width=2100, height=2100)
    camera_id = "oversized_decoded_frame"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/frame",
            files={"image": ("large.jpg", jpeg, "image/jpeg")},
            data={"camera_id": camera_id},
        )

    assert response.status_code == 413
    assert app.state.latest_frame_store.get(camera_id) is None
    assert camera_id not in app.state.camera_registry
    app.state.detector.detect.assert_not_called()


@pytest.mark.asyncio
async def test_phone_can_omit_frame_echo_without_removing_websocket_frame():
    class RecordingManager:
        def __init__(self):
            self.messages = []

        async def broadcast_json(self, data):
            self.messages.append(data)

    manager = RecordingManager()
    app.state.ws_manager = manager
    jpeg = _make_test_jpeg()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/frame",
            files={"image": ("frame.jpg", jpeg, "image/jpeg")},
            data={
                "camera_id": "phone",
                "timestamp": "1000.0",
                "include_frame": "false",
            },
        )

    assert resp.status_code == 200
    assert resp.json()["frame_jpeg_b64"] == ""
    assert len(manager.messages) == 1
    assert base64.b64decode(manager.messages[0]["frame_jpeg_b64"]) == jpeg


@pytest.mark.asyncio
async def test_frame_disables_metric_projection_after_camera_aspect_changes():
    from backend.calibration import calibrate_from_markers
    from backend.marker_detector import MarkerDetection
    from backend.zones_store import Zone

    def marker(marker_id, x, y, half=20):
        return MarkerDetection(
            marker_id=marker_id,
            corners=[
                (x - half, y - half),
                (x + half, y - half),
                (x + half, y + half),
                (x - half, y + half),
            ],
        )

    camera_id = "cam_phone_rotates"
    app.state.zone_store.replace(camera_id, [
        Zone(
            name="Test",
            polygon=[[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
        ),
    ])
    app.state.calibration_store.set(calibrate_from_markers(
        camera_id=camera_id,
        detections=[
            marker(10, 64, 48),
            marker(20, 576, 48),
            marker(30, 576, 432),
            marker(40, 64, 432),
        ],
        marker_ids=[10, 20, 30, 40],
        width_m=6.4,
        height_m=4.8,
        frame_width=640,
        frame_height=480,
    ))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        landscape = await client.post(
            "/api/frame",
            files={
                "image": (
                    "landscape.jpg",
                    _make_test_jpeg(width=640, height=480),
                    "image/jpeg",
                ),
            },
            data={"camera_id": camera_id, "timestamp": "1000.0"},
        )
        portrait = await client.post(
            "/api/frame",
            files={
                "image": (
                    "portrait.jpg",
                    _make_test_jpeg(width=480, height=640),
                    "image/jpeg",
                ),
            },
            data={"camera_id": camera_id, "timestamp": "1001.0"},
        )

    landscape_data = landscape.json()
    assert landscape_data["calibration_active"] is True
    assert landscape_data["calibration_valid"] is True
    assert landscape_data["calibration_warning"] is None
    assert all(
        zone["calibrated"]
        for zone in landscape_data["dynamic_safety_zones"]
    )

    portrait_data = portrait.json()
    assert portrait_data["calibration_active"] is True
    assert portrait_data["calibration_valid"] is False
    assert "aspect mismatch" in portrait_data["calibration_warning"].lower()
    assert all(
        not zone["calibrated"]
        for zone in portrait_data["dynamic_safety_zones"]
    )
    assert portrait_data["person_distances"]
    assert all(
        distance["distance_m"] is None
        for distance in portrait_data["person_distances"]
    )


@pytest.mark.asyncio
async def test_non_jpeg_upload_is_normalized_to_raw_jpeg_preview():
    frame = np.full((120, 160, 3), 91, dtype=np.uint8)
    ok, png = cv2.imencode(".png", frame)
    assert ok

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/frame",
            files={"image": ("frame.png", png.tobytes(), "image/png")},
            data={"camera_id": "test", "timestamp": "1000.0"},
        )

    assert resp.status_code == 200
    preview = base64.b64decode(resp.json()["frame_jpeg_b64"])
    assert preview.startswith(b"\xff\xd8")
    decoded = cv2.imdecode(np.frombuffer(preview, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == frame.shape
    # Detector boxes are metadata only; a uniform source stays uniform.
    assert int(decoded.max()) - int(decoded.min()) <= 3


@pytest.mark.asyncio
async def test_alert_evidence_stays_annotated_while_live_preview_is_raw():
    detector = _mock_detector()
    detector.detect.return_value = [
        Detection(class_id=0, class_name="person", category="person",
                  box=(30, 25, 90, 105), confidence=0.85),
        Detection(class_id=7, class_name="truck", category="vehicle",
                  box=(20, 20, 115, 110), confidence=0.92),
    ]
    app.state.detector = detector
    source = np.full((140, 180, 3), 90, dtype=np.uint8)
    ok, jpeg_buffer = cv2.imencode(".jpg", source)
    assert ok
    jpeg = jpeg_buffer.tobytes()

    confirmed = None
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for index in range(CONFIG.danger.consecutive_frames_required):
            resp = await client.post(
                "/api/frame",
                files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                data={"camera_id": "test", "timestamp": str(1000.0 + index)},
            )
            body = resp.json()
            assert base64.b64decode(body["frame_jpeg_b64"]) == jpeg
            if body["confirmed_alerts"]:
                confirmed = body["confirmed_alerts"][0]

    assert confirmed is not None
    thumbnail_name = Path(confirmed["frame_thumbnail_url"]).name
    evidence = cv2.imread(str(Path(app.state.frame_store.base_dir) / thumbnail_name))
    assert evidence is not None
    # Saved evidence includes high-contrast detection/alarm graphics, unlike
    # the uniform live background delivered above.
    assert int(evidence.max()) - int(evidence.min()) > 80


@pytest.mark.asyncio
async def test_post_frame_no_dangers_far_apart():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        jpeg = _make_test_jpeg()
        resp = await client.post(
            "/api/frame",
            files={"image": ("frame.jpg", jpeg, "image/jpeg")},
            data={"camera_id": "test", "timestamp": "1000.0"},
        )
        data = resp.json()
        assert len(data["active_dangers"]) == 0
        assert len(data["confirmed_alerts"]) == 0


@pytest.mark.asyncio
async def test_post_frame_with_overlap():
    det = _mock_detector()
    det.detect.return_value = [
        Detection(class_id=0, class_name="person", category="person",
                  box=(150, 150, 250, 300), confidence=0.85),
        Detection(class_id=7, class_name="truck", category="vehicle",
                  box=(100, 100, 300, 300), confidence=0.92),
    ]
    app.state.detector = det

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        jpeg = _make_test_jpeg()
        resp = await client.post(
            "/api/frame",
            files={"image": ("frame.jpg", jpeg, "image/jpeg")},
            data={"camera_id": "test", "timestamp": "1000.0"},
        )
        data = resp.json()
        assert len(data["active_dangers"]) > 0


@pytest.mark.asyncio
async def test_alerts_empty():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/alerts")
        assert resp.status_code == 200
        assert resp.json() == []


@pytest.mark.asyncio
async def test_frame_counter_increments():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        jpeg = _make_test_jpeg()
        for i in range(3):
            resp = await client.post(
                "/api/frame",
                files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                data={"camera_id": "test"},
            )
            assert resp.json()["frame_id"] == i + 1

@pytest.mark.asyncio
async def test_post_frame_includes_confirmed_posture_alert():
    from backend.posture_detector import PostureAssessment, PostureProcessResult

    class FakePostureManager:
        available = True
        unavailable_reason = None

        def process(self, camera_id, frame, persons, timestamp):
            assessment = PostureAssessment(
                track_id=7,
                person=persons[0],
                risk_score=0.74,
                severity="WARNING",
                status="verification_required",
                signals=["repeated_body_sway", "unstable_trajectory"],
                metrics={"torso_sway_deg": 9.2},
                frame_timestamp=timestamp,
                pose_confidence=0.91,
                history_seconds=2.8,
                confirmed=True,
                landmarks=None,
            )
            return PostureProcessResult([assessment], inference_ran=True)

    old_manager = getattr(app.state, "posture_manager", None)
    app.state.posture_manager = FakePostureManager()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/frame",
                files={"image": ("frame.jpg", _make_test_jpeg(), "image/jpeg")},
                data={"camera_id": "test", "timestamp": "1000.0"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["posture_available"] is True
        assert len(data["posture_assessments"]) == 1
        assert len(data["confirmed_posture_alerts"]) == 1
        alert = data["confirmed_posture_alerts"][0]
        assert alert["track_id"] == 7
        assert alert["risk_score"] == 0.74
        assert app.state.alert_store.query(kind="posture_anomaly")[0].rule_name == "coordination_anomaly"
    finally:
        app.state.posture_manager = old_manager

@pytest.mark.asyncio
async def test_possible_fall_is_saved_as_dedicated_event():
    from backend.posture_detector import PostureAssessment, PostureProcessResult

    class FakeFallManager:
        available = True
        unavailable_reason = None

        def process(self, camera_id, frame, persons, timestamp):
            return PostureProcessResult([
                PostureAssessment(
                    track_id=11,
                    person=persons[0],
                    risk_score=0.93,
                    severity="DANGER",
                    status="high_risk",
                    signals=["sudden_balance_loss", "possible_fall"],
                    metrics={"horizontal_torso_fraction": 0.7},
                    frame_timestamp=timestamp,
                    pose_confidence=0.90,
                    history_seconds=2.4,
                    confirmed=True,
                    landmarks=None,
                )
            ], inference_ran=True)

    old_manager = getattr(app.state, "posture_manager", None)
    app.state.posture_manager = FakeFallManager()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/frame",
                files={"image": ("frame.jpg", _make_test_jpeg(), "image/jpeg")},
                data={"camera_id": "test", "timestamp": "1000.0"},
            )
        assert resp.status_code == 200
        fall = resp.json()["confirmed_posture_alerts"][0]
        assert "possible_fall" in fall["signals"]
        stored = app.state.alert_store.query(kind="fall_detected")
        assert len(stored) == 1
        assert stored[0].rule_name == "possible_fall"
    finally:
        app.state.posture_manager = old_manager

@pytest.mark.asyncio
async def test_hand_to_mouth_candidate_is_saved_as_smoking_gesture():
    from backend.posture_detector import PostureAssessment, PostureProcessResult

    class FakeGestureManager:
        available = True
        unavailable_reason = None

        def process(self, camera_id, frame, persons, timestamp):
            return PostureProcessResult([
                PostureAssessment(
                    track_id=12,
                    person=persons[0],
                    risk_score=0.55,
                    severity="WARNING",
                    status="verification_required",
                    signals=["hand_to_mouth_pattern"],
                    metrics={"hand_to_mouth_fraction": 0.75},
                    frame_timestamp=timestamp,
                    pose_confidence=0.91,
                    history_seconds=2.2,
                    confirmed=True,
                    landmarks=None,
                )
            ], inference_ran=True)

    old_manager = getattr(app.state, "posture_manager", None)
    app.state.posture_manager = FakeGestureManager()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/frame",
                files={"image": ("frame.jpg", _make_test_jpeg(), "image/jpeg")},
                data={"camera_id": "test", "timestamp": "1000.0"},
            )
        assert resp.status_code == 200
        stored = app.state.alert_store.query(kind="smoking_gesture")
        assert len(stored) == 1
        assert stored[0].rule_name == "hand_to_mouth_pattern"
        assert stored[0].details["interpretation"] == "requires_human_verification"
    finally:
        app.state.posture_manager = old_manager


@pytest.mark.asyncio
async def test_camera_registry_and_frame_camera_id():
    app.state.camera_registry = {}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        jpeg = _make_test_jpeg()
        resp = await client.post(
            "/api/frame",
            files={"image": ("frame.jpg", jpeg, "image/jpeg")},
            data={"camera_id": "cam_north", "timestamp": "2500.0"},
        )
        assert resp.status_code == 200
        assert resp.json()["camera_id"] == "cam_north"
        cameras = (await client.get("/api/cameras")).json()["cameras"]
        assert cameras[0]["camera_id"] == "cam_north"
        assert cameras[0]["width"] > 0
        assert cameras[0]["captured_at"] == 2500.0
        assert cameras[0]["last_seen"] > 2500.0
        assert cameras[0]["online"] is True

@pytest.mark.asyncio
async def test_readiness_reports_demo_components(tmp_path):
    from backend.evidence import EvidenceRecorder
    from config import EvidenceConfig

    app.state.evidence_recorder = EvidenceRecorder(
        EvidenceConfig(enabled=True, clips_dir=str(tmp_path / "clips"))
    )
    app.state.camera_registry = {}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/readiness")
        assert resp.status_code == 200
        data = resp.json()
        assert data["components"]["browser_video_demo"] is True
        assert data["components"]["operator_review"] is True
        assert "ready_for_metric_demo" in data
    app.state.evidence_recorder.close()


@pytest.mark.asyncio
async def test_readiness_requires_one_online_camera_with_matching_zone_and_aspect(
    tmp_path,
):
    from backend.calibration import calibrate_from_markers
    from backend.evidence import EvidenceRecorder
    from backend.marker_detector import MarkerDetection
    from backend.zones_store import Zone
    from config import EvidenceConfig

    def marker(marker_id, center_x, center_y, size=36):
        return MarkerDetection(
            marker_id=marker_id,
            corners=[
                (center_x - size, center_y - size),
                (center_x + size, center_y - size),
                (center_x + size, center_y + size),
                (center_x - size, center_y + size),
            ],
        )

    camera_id = "cam_phone_1"
    app.state.evidence_recorder = EvidenceRecorder(
        EvidenceConfig(enabled=True, clips_dir=str(tmp_path / "clips-metric"))
    )
    app.state.camera_registry = {
        camera_id: {
            "camera_id": camera_id,
            "last_seen": time.time(),
            "mode": "site",
            "width": 960,
            "height": 720,
        },
    }
    app.state.zone_store.replace(camera_id, [
        Zone(
            name="Strefa telefonu",
            polygon=[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        ),
    ])
    app.state.calibration_store.set(calibrate_from_markers(
        camera_id=camera_id,
        detections=[
            marker(10, 192, 144),
            marker(20, 768, 144),
            marker(30, 768, 576),
            marker(40, 192, 576),
        ],
        marker_ids=[10, 20, 30, 40],
        width_m=3.0,
        height_m=2.0,
        frame_width=960,
        frame_height=720,
    ))

    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            ready = (await client.get("/api/readiness")).json()
            row = next(
                item for item in ready["cameras"]
                if item["camera_id"] == camera_id
            )
            assert row["online"] is True
            assert row["zones_configured"] is True
            assert row["calibration_exists"] is True
            assert row["calibration_valid"] is True
            assert row["metric_distance_ready"] is True
            assert ready["ready_for_metric_demo"] is True

            # Rotating 4:3 phone output to 3:4 must disable metric use.
            app.state.camera_registry[camera_id]["width"] = 720
            app.state.camera_registry[camera_id]["height"] = 960
            incompatible = (await client.get("/api/readiness")).json()
            row = next(
                item for item in incompatible["cameras"]
                if item["camera_id"] == camera_id
            )
            assert row["calibration_exists"] is True
            assert row["calibration_valid"] is False
            assert row["metric_distance_ready"] is False
            assert "aspect" in row["calibration_warning"].lower()
            assert incompatible["ready_for_metric_demo"] is False
    finally:
        app.state.evidence_recorder.close()

@pytest.mark.asyncio
async def test_camera_registry_includes_offline_calibrated_camera(tmp_path):
    from backend.calibration import Calibration, CalibrationStore

    previous_store = app.state.calibration_store
    previous_registry = app.state.camera_registry
    store = CalibrationStore(str(tmp_path / "calibration.json"))
    store.set(Calibration(
        camera_id="cam_saved",
        marker_ids=[10, 20, 30, 40],
        width_m=3.0,
        height_m=2.0,
        homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        source_frame_width=960,
        source_frame_height=720,
        source_aspect_ratio=4 / 3,
    ))
    app.state.calibration_store = store
    app.state.camera_registry = {}
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/cameras")
        assert response.status_code == 200
        rows = response.json()["cameras"]
        row = next(item for item in rows if item["camera_id"] == "cam_saved")
        assert row["online"] is False
        assert row["calibrated"] is True
        assert row["calibration_created_at"] is not None
    finally:
        app.state.calibration_store = previous_store
        app.state.camera_registry = previous_registry


@pytest.mark.asyncio
async def test_async_frame_post_returns_202_without_inline_detector_and_reports_replacement():
    from backend.frame_processor import (
        FrameProcessorStats,
        FrameSubmitResult,
        FrameSubmitStatus,
    )

    class FakeFrameProcessor:
        def __init__(self):
            self.jobs = []

        async def submit(self, job):
            self.jobs.append(job)
            status = (
                FrameSubmitStatus.ACCEPTED
                if len(self.jobs) == 1
                else FrameSubmitStatus.REPLACED
            )
            return FrameSubmitResult(
                camera_id=job.camera_id,
                status=status,
                queue_depth=1,
            )

        @property
        def stats(self):
            return FrameProcessorStats(
                submitted=len(self.jobs),
                replaced=max(0, len(self.jobs) - 1),
            )

    previous_processor = getattr(app.state, "frame_processor", None)
    fake_processor = FakeFrameProcessor()
    app.state.frame_processor = fake_processor
    app.state.detector.detect.reset_mock()
    jpeg = _make_test_jpeg()

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.post(
                "/api/frame",
                files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                data={
                    "camera_id": "cam_async_fake",
                    "timestamp": "101.0",
                    "async_processing": "true",
                },
            )
            second = await client.post(
                "/api/frame",
                files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                data={
                    "camera_id": "cam_async_fake",
                    "timestamp": "102.0",
                    "async_processing": "true",
                },
            )

        assert first.status_code == 202
        assert first.json() == {
            "accepted": True,
            "camera_id": "cam_async_fake",
            "timestamp": 101.0,
            "mode": "site",
            "queue_depth": 1,
            "replaced_pending": False,
            "dropped_frames": 0,
        }
        assert second.status_code == 202
        assert second.json() == {
            "accepted": True,
            "camera_id": "cam_async_fake",
            "timestamp": 102.0,
            "mode": "site",
            "queue_depth": 1,
            "replaced_pending": True,
            "dropped_frames": 1,
        }
        assert [job.timestamp for job in fake_processor.jobs] == [101.0, 102.0]
        app.state.detector.detect.assert_not_called()

        registry = app.state.camera_registry["cam_async_fake"]
        assert registry["frames_received"] == 2
        assert registry["captured_at"] == 102.0
        assert registry["processing_pending"] is True
    finally:
        app.state.frame_processor = previous_processor


@pytest.mark.asyncio
async def test_real_latest_frame_processor_drain_publishes_and_updates_registry():
    from backend.frame_processor import LatestFrameProcessor
    from backend.routes import ingest

    class RecordingConnectionManager:
        def __init__(self):
            self.frames = []

        async def broadcast_frame(self, data, jpeg_bytes=None):
            self.frames.append((data, jpeg_bytes))

    previous_processor = getattr(app.state, "frame_processor", None)
    previous_ws_manager = app.state.ws_manager
    optional_state = {
        name: getattr(app.state, name, None)
        for name in (
            "evidence_recorder",
            "posture_manager",
            "posture_worker",
            "worker_identifier",
            "unidentified_worker_monitor",
            "worker_store",
            "marker_scheduler",
        )
    }
    recorder = RecordingConnectionManager()
    app.state.ws_manager = recorder
    app.state.evidence_recorder = None
    app.state.posture_manager = None
    app.state.posture_worker = None
    app.state.worker_identifier = None
    app.state.unidentified_worker_monitor = None
    app.state.worker_store = None
    app.state.marker_scheduler = None
    app.state.detector.detect.reset_mock()

    processor = LatestFrameProcessor(
        process=lambda job: ingest.process_frame_job(app, job),
        publish=lambda job, result: ingest.publish_frame_result(app, job, result),
        max_cameras=2,
    )
    await processor.start()
    app.state.frame_processor = processor
    jpeg = _make_test_jpeg()

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/frame",
                files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                data={
                    "camera_id": "cam_async_real",
                    "timestamp": "4321.0",
                    "mode": "site",
                    "async_processing": "true",
                },
            )

        assert response.status_code == 202
        assert response.json()["accepted"] is True
        assert response.json()["camera_id"] == "cam_async_real"

        await processor.drain()

        assert processor.stats.processed == 1
        assert processor.stats.failed == 0
        assert app.state.detector.detect.call_count == 1
        assert len(recorder.frames) == 1
        published, published_jpeg = recorder.frames[0]
        assert published["camera_id"] == "cam_async_real"
        assert published["timestamp"] == 4321.0
        assert len(published["detections"]) == 2
        assert published_jpeg == jpeg

        registry = app.state.camera_registry["cam_async_real"]
        assert registry["frames_received"] == 1
        assert registry["frames"] == 1
        assert registry["captured_at"] == 4321.0
        assert registry["processed_captured_at"] == 4321.0
        assert registry["processed_mode"] == "site"
        assert registry["processing_pending"] is False
        assert registry["last_processed_received_at"] == registry["last_seen"]
        assert registry["width"] == 640
        assert registry["height"] == 480

        latest = app.state.latest_frame_store.get("cam_async_real")
        assert latest is not None
        assert latest.timestamp == 4321.0
        assert latest.image_bytes == jpeg
    finally:
        await processor.close()
        app.state.frame_processor = previous_processor
        app.state.ws_manager = previous_ws_manager
        for name, value in optional_state.items():
            setattr(app.state, name, value)
