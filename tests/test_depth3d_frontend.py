from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend" / "depth3d.html").read_text(encoding="utf-8")
JS = (ROOT / "frontend" / "depth3d.js").read_text(encoding="utf-8")


def test_depth3d_page_uses_existing_camera_registry_and_no_browser_webcam():
    assert 'id="cameraSelect"' in HTML
    assert 'api("/api/cameras")' in JS
    assert "getUserMedia" not in HTML
    assert "getUserMedia" not in JS


def test_depth3d_v2_uses_checkerboard_single_point_depth_and_fps():
    for endpoint in ("/checkerboard.png", "/checkerboard/detect", "/checkerboard/capture", "/checkerboard/calibrate", "/infer", "/distance"):
        assert endpoint in JS
    assert 'id="cornersX"' in HTML
    assert 'id="depthCanvas"' in HTML
    assert 'id="modelFpsMeta"' in HTML
    assert 'id="streamFpsMeta"' in HTML
    assert "ChArUco" not in HTML
    assert "points_f32_b64" not in JS


def test_realtime_loop_subtracts_inference_time_from_target_interval():
    assert "Math.max(0, targetIntervalMs - elapsed)" in JS
    assert "displayFpsEma" in JS


def test_depth3d_v21_draws_person_distance_overlays_and_uses_jpeg_depth():
    assert 'id="peopleResult"' in HTML
    assert 'id="personMeta"' in HTML
    assert 'drawPersonOverlays' in JS
    assert 'person_distances' in JS
    assert 'depth_jpeg_b64' in JS
