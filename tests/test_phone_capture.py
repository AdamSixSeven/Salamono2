import re
import shutil
import subprocess
from pathlib import Path

import pytest


CAPTURE_PATH = Path(__file__).resolve().parents[1] / "phone" / "capture.html"
CAPTURE_HTML = CAPTURE_PATH.read_text(encoding="utf-8")
SCRIPT_MATCH = re.search(
    r"<script>\s*(\(function\(\) \{[\s\S]*?\}\)\(\);)\s*</script>",
    CAPTURE_HTML,
)
assert SCRIPT_MATCH is not None
SCRIPT = SCRIPT_MATCH.group(1)


def function_source(name: str) -> str:
    start = SCRIPT.index(f"function {name}(")
    next_function = SCRIPT.find("\n        function ", start + 1)
    if next_function == -1:
        return SCRIPT[start:]
    return SCRIPT[start:next_function]


def test_phone_capture_exposes_all_supported_frame_rates():
    fps_select = re.search(
        r'<select id="fpsSelect">(.*?)</select>',
        CAPTURE_HTML,
        flags=re.DOTALL,
    )
    assert fps_select is not None

    options = re.findall(
        r'<option value="(\d+)"([^>]*)>',
        fps_select.group(1),
    )
    assert [int(value) for value, _ in options] == [5, 8, 15, 22, 30]
    assert [int(value) for value, attrs in options if "selected" in attrs] == [5]


def test_phone_capture_measures_the_complete_cycle_and_keeps_backpressure():
    cycle_start = CAPTURE_HTML.index("var cycleStartedAt = performance.now();")
    jpeg_encoding = CAPTURE_HTML.index("captureCanvas.toBlob(", cycle_start)
    request_start = CAPTURE_HTML.index("fetch(serverUrl + \"/api/frame\"", jpeg_encoding)

    assert cycle_start < jpeg_encoding < request_start
    assert "if (inFlight)" in CAPTURE_HTML
    assert "inFlight = true;" in CAPTURE_HTML
    assert "var parsedFps = parseInt(fpsSelect.value, 10);" in CAPTURE_HTML
    assert "var targetInterval = 1000 / fps;" in CAPTURE_HTML
    assert "var cycleElapsed = performance.now() - cycleStartedAt;" in CAPTURE_HTML
    assert "Math.max(0, targetInterval - cycleElapsed)" in CAPTURE_HTML
    assert "inFlight = false;" in CAPTURE_HTML


def test_phone_requests_metadata_only_but_still_parses_analysis_response():
    socket = function_source("connectAnalysisSocket")
    consume = function_source("consumeAnalysis")
    start = function_source("startCapture")
    stop = function_source("stopCapture")

    assert 'target.searchParams.set("camera_id", streamCameraId())' in socket
    assert 'target.searchParams.set("include_frame", "false")' in socket
    assert 'typeof event.data !== "string"' in socket
    assert "consumeAnalysis(data)" in socket
    assert '(data.camera_id || "cam_default") !== streamCameraId()' in consume
    assert "announceAlerts(data);" in consume
    assert "data.processing_ms || 0" in consume
    assert "connectAnalysisSocket()" in start
    assert "stopAnalysisSocket()" in stop


def test_phone_enables_latest_wins_only_with_metadata_socket_fallback():
    send = function_source("sendOneFrame")

    assert 'form.append("camera_id", cameraId)' in send
    assert 'form.append("include_frame", "false")' in send
    assert '"async_processing"' in send
    assert 'analysisSocketConnected ? "true" : "false"' in send
    assert "return resp.json()" in send
    assert "data && data.accepted" in send
    assert "data.queue_depth || 0" in send
    assert "data.dropped_frames || 0" in send
    assert "consumeAnalysis(data)" in send


def test_phone_inline_javascript_syntax():
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


def test_phone_exposes_checkerboard_capture_for_depth_calibration():
    assert 'id="depthCaptureBtn"' in CAPTURE_HTML
    assert 'id="depthCornersX"' in CAPTURE_HTML
    assert 'id="depthCornersY"' in CAPTURE_HTML
    assert 'id="depthSquareMm"' in CAPTURE_HTML
    assert '"/checkerboard/capture"' in CAPTURE_HTML
    assert '"/api/depth3d/checkerboard.png?"' in CAPTURE_HTML
