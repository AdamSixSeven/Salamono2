"""Static/browser-contract tests for the calibration page.

These tests intentionally do not duplicate calibration API behaviour covered
by ``test_calibration_api.py``.  They protect the browser-side source,
selection, freshness and payload contracts that can otherwise regress while
the backend remains perfectly valid.
"""

from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend" / "calibrate.html").read_text(encoding="utf-8")
APP_PATH = ROOT / "frontend" / "app.js"
APP = APP_PATH.read_text(encoding="utf-8")
SCRIPT_MATCH = re.search(
    r"<script>\s*(\(function \(\) \{[\s\S]*?\}\)\(\);)\s*</script>",
    HTML,
)
assert SCRIPT_MATCH is not None
SCRIPT = SCRIPT_MATCH.group(1)


def source_function(source: str, name: str) -> str:
    start = source.index(f"function {name}(")
    next_function = source.find("\n    function ", start + 1)
    if next_function == -1:
        return source[start:]
    return source[start:next_function]


def function_source(name: str) -> str:
    return source_function(SCRIPT, name)


def test_active_camera_selection_has_required_priority_and_registry_source():
    source = function_source("chooseInitialCamera")

    assert 'fetch("/api/cameras"' in SCRIPT
    assert source.index("queryCameraId()") < source.index("storedCameraId()")
    assert source.index("storedCameraId()") < source.index("cameras.find")
    assert source.index("cameras.find") < source.index("display fallback only")


def test_temporary_cam_default_does_not_block_first_online_camera():
    choose = function_source("chooseInitialCamera")
    refresh = function_source("refreshCameras")
    apply = function_source("applyCameraId")
    calibration = function_source("refreshCalibration")

    assert "storedExists" in choose
    assert "cameras.some" in choose
    assert 'fromUrl !== "cam_default" || urlCameraExists' in choose
    fallback = choose[choose.index("display fallback only"):]
    assert 'cameraId: "cam_default"' in fallback
    assert "locked: false" in fallback
    assert "remember: false" in fallback
    assert "else if (!State.selectionLocked)" in refresh
    assert 'source: "first-online"' in refresh
    assert "remember: true" in refresh
    assert "if (options.remember === true)" in apply
    assert 'previousSource === "auto-fallback"' in apply
    assert "changed || placeholderResolved" in apply
    assert 'State.selectionSource === "auto-fallback"' in calibration
    assert "!cameraById(requestedCamera)" in calibration
    assert 'source: "manual"' in SCRIPT
    assert "locked: true" in SCRIPT[
        SCRIPT.index('dom.cameraSelect.addEventListener("change"'):
    ]


def test_remote_preview_uses_filtered_binary_websocket_with_base64_fallback():
    receive = function_source("receiveLiveFrame")
    receive_blob = function_source("receiveLiveFrameBlob")
    socket = function_source("connectStream")

    assert '"/ws/live"' in socket
    assert '"?camera_id=" + encodeURIComponent(State.cameraId) + "&binary=true"' in socket
    assert 'socket.binaryType = "blob"' in socket
    assert '(data.camera_id || "cam_default") !== State.cameraId' in socket
    assert 'data.frame_transport === "binary-jpeg"' in socket
    assert "State.pendingBinaryFrame = data" in socket
    assert "receiveLiveFrameBlob(metadata, event.data)" in socket
    assert "createImageBitmap(blob)" in receive_blob
    assert 'image.src = "data:image/jpeg;base64," + data.frame_jpeg_b64' in receive
    assert "/latest-frame" not in SCRIPT
    assert "streamPreview" not in HTML
    assert "sourceMode" not in HTML
    # The metadata preview endpoint must never become an image source.
    assert "analysis.frame_jpeg_b64" not in SCRIPT


def test_canvas_uses_decoded_frame_dimensions_and_real_aspect_ratio():
    receive = function_source("receiveLiveFrame")
    receive_blob = function_source("receiveLiveFrameBlob")
    apply_image = function_source("applyRemoteImage")

    assert "image.naturalWidth || image.width" in receive
    assert "image.naturalHeight || image.height" in receive
    assert "bitmap.width" in receive_blob
    assert "bitmap.height" in receive_blob
    assert "generation !== State.imageGeneration" in apply_image
    assert "data.camera_id !== State.cameraId" in apply_image
    assert "dom.previewCanvas.width = width" in apply_image
    assert "dom.previewCanvas.height = height" in apply_image
    assert 'dom.previewShell.style.aspectRatio = width + " / " + height' in apply_image


def test_main_panel_pairs_binary_jpeg_with_filtered_metadata():
    connect = source_function(APP, "connectWebSocket")
    message = source_function(APP, "onWsMessage")
    render_binary = source_function(APP, "renderBinaryFrame")
    apply_frame = source_function(APP, "applyDecodedFrame")
    select_camera = source_function(APP, "selectCamera")

    assert '"?camera_id=" + encodeURIComponent(State.cameraId) + "&binary=true"' in connect
    assert 'socket.binaryType = "blob"' in connect
    assert '(data.camera_id || "cam_default") !== State.cameraId' in message
    assert 'data.frame_transport === "binary-jpeg"' in message
    assert "pendingBinaryFrames.push(data)" in message
    assert "pendingBinaryFrames.shift()" in message
    assert "createImageBitmap(blob)" in render_binary
    assert "bitmap.width" in render_binary
    assert "bitmap.height" in render_binary
    assert 'typeof image.close === "function"' in apply_frame
    assert "pendingBinaryFrames.length = 0" in select_camera


def test_camera_and_preview_metadata_are_visible_and_preview_api_is_used():
    for element_id in (
        "cameraOnline",
        "socketStatus",
        "cameraMode",
        "cameraLastSeen",
        "cameraResolution",
        "frameFreshness",
        "detectedMarkerIds",
        "missingMarkerIds",
        "previewValidity",
        "calibrationFlag",
    ):
        assert f'id="{element_id}"' in HTML

    assert '"/preview?marker_ids="' in SCRIPT
    assert "State.previewAnalysis.detected_markers" in SCRIPT
    assert "analysis.validation_error" in SCRIPT
    assert "analysis.frame_available" in SCRIPT
    assert "analysis.frame_age_sec" in SCRIPT
    assert "analysis.fresh" in SCRIPT
    assert "analysis.calibration_exists" in SCRIPT
    assert "analysis.calibration_valid_for_frame" in SCRIPT
    assert "analysis.calibration_active" in SCRIPT
    assert "analysis.calibration_aspect_compatible" in SCRIPT


def test_latest_frame_calibration_has_exact_guard_and_json_contract():
    readiness = function_source("updateReadiness")
    marker_guard = function_source("requiredMarkersReady")
    submit = function_source("calibrateFromLatest")

    assert "FRAME_FRESH_MS = 3000" in SCRIPT
    assert (
        "online && fresh && idsOk && dimensionsOk && !State.calibrationBusy"
        in readiness
    )
    assert "State.previewAnalysis.frame_available === true" in readiness
    assert "State.previewAnalysis.fresh === true" in readiness
    assert "analysis.camera_id !== State.cameraId" in marker_guard
    assert "analysis.valid !== true" in marker_guard
    assert "analysis.required_marker_ids" in marker_guard
    assert "analysis.detected_markers" in marker_guard
    assert "requiredIds.every" in marker_guard
    assert '"4/4 wymaganych markerów wykryte"' in readiness
    assert '"/from-latest"' in submit
    assert '"Content-Type": "application/json"' in submit
    assert "marker_ids: ids" in submit
    assert "width_m: values.widthM" in submit
    assert "height_m: values.heightM" in submit
    assert "new Set(ids).size === 4" in SCRIPT
    assert "value >= 50" in SCRIPT


def test_quality_result_contract_is_rendered():
    quality = function_source("renderQuality")

    for field in (
        "status",
        "score",
        "score_percent",
        "coverage_ratio",
        "quadrilateral_area_ratio",
        "min_edge_ratio",
        "smallest_marker_area_px",
        "min_angle_deg",
        "max_angle_deg",
        "reprojection_error_normalized",
        "condition_number",
        "warnings",
    ):
        assert field in quality


def test_two_click_measurement_uses_exact_api_payload_once():
    submit = function_source("submitMeasurement")
    measurement_guard = function_source("calibrationReadyForMeasurement")

    assert SCRIPT.count(
        'dom.previewCanvas.addEventListener("click", handleMeasureCanvasClick)'
    ) == 1
    assert '"/measure"' in submit
    assert "point_a: [points[0][0], points[0][1]]" in submit
    assert "point_b: [points[1][0], points[1][1]]" in submit
    assert "const frameWidth = dom.previewCanvas.width" in submit
    assert "const frameHeight = dom.previewCanvas.height" in submit
    assert "frame_width: frameWidth" in submit
    assert "frame_height: frameHeight" in submit
    assert "State.measurePoints.length === 2" in SCRIPT
    assert "analysis.calibration_valid_for_frame === true" in measurement_guard
    assert "analysis.camera_id === State.cameraId" in measurement_guard
    assert "requestedCamera !== State.cameraId" in submit


def test_async_camera_mutations_ignore_results_after_selection_changes():
    calibrate = function_source("calibrateFromLatest")
    clear = function_source("clearCalibration")
    measure = function_source("submitMeasurement")

    for source in (calibrate, clear, measure):
        assert "const requestedCamera = State.cameraId" in source
        assert "encodeURIComponent(requestedCamera)" in source
        assert "requestedCamera !== State.cameraId" in source


def test_calibration_uses_registered_cameras_only_and_auto_connects():
    apply_camera = function_source("applyCameraId")
    refresh = function_source("refreshCameras")
    auto_connect = function_source("autoConnectSelectedCamera")
    select_binding = SCRIPT[
        SCRIPT.index('dom.cameraSelect.addEventListener("change"'):
    ]

    assert 'id="cameraSelect"' in HTML
    assert 'fetch("/api/cameras"' in SCRIPT
    assert "navigator.mediaDevices.getUserMedia" not in SCRIPT
    assert "navigator.mediaDevices.enumerateDevices" not in SCRIPT
    assert 'id="localCameraDetails"' not in HTML
    assert 'id="localPreview"' not in HTML
    assert "window.setTimeout(autoConnectSelectedCamera, 0)" in apply_camera
    assert "autoConnectSelectedCamera();" in refresh
    assert "connectStream(false);" in auto_connect
    assert "autoConnectSelectedCamera();" in select_binding
    assert "skalibrowana" in function_source("renderCameraOptions")


def test_marker_layout_and_png_downloads_cover_default_ids():
    expected_positions = {
        10: "lewy górny · TL",
        20: "prawy górny · TR",
        30: "prawy dolny · BR",
        40: "lewy dolny · BL",
    }
    for marker_id, position in expected_positions.items():
        assert position in HTML
        url = f"/api/calibration/markers/{marker_id}.png"
        assert HTML.count(url) >= 2  # preview image + download link
        assert f'download="aruco-{marker_id}.png"' in HTML
    assert "window.print()" in SCRIPT


def test_event_binding_is_centralized_and_not_called_twice():
    assert SCRIPT.count("function bindEventsOnce()") == 1
    assert SCRIPT.count("bindEventsOnce();") == 1
    for binding in (
        'dom.connectStreamBtn.addEventListener("click", function ()',
        'dom.calibrateLatestBtn.addEventListener("click", calibrateFromLatest)',
        'dom.cameraSelect.addEventListener("change", function ()',
    ):
        assert SCRIPT.count(binding) == 1


def test_inline_javascript_syntax():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    result = subprocess.run(
        [node, "--check", "-"],
        input=SCRIPT.encode("utf-8"),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")


def test_main_panel_javascript_syntax():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    result = subprocess.run(
        [node, "--check", str(APP_PATH)],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
