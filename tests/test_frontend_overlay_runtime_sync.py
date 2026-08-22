from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "frontend" / "app.js"
ZONES_PATH = ROOT / "frontend" / "zones.js"
APP = APP_PATH.read_text(encoding="utf-8")
ZONES = ZONES_PATH.read_text(encoding="utf-8")


def source_function(source: str, name: str) -> str:
    start = source.index(f"function {name}(")
    next_function = source.find("\n    function ", start + 1)
    if next_function == -1:
        return source[start:]
    return source[start:next_function]


def test_decoded_frame_keeps_canvas_buffer_when_resolution_is_unchanged():
    apply_frame = source_function(APP, "applyDecodedFrame")

    assert "if (canvas.width !== width || canvas.height !== height)" in apply_frame
    assert apply_frame.count("canvas.width = width;") == 1
    assert apply_frame.count("canvas.height = height;") == 1
    assert "ctx.drawImage(image, 0, 0);" in apply_frame
    assert 'new CustomEvent("perimetr-frame-rendered"' in apply_frame


def test_overlay_has_one_authoritative_frame_render_path_and_metadata_fallback():
    rendered = source_function(ZONES, "renderSpatialFrame")
    frame_key = source_function(ZONES, "spatialFrameKey")
    fallback = source_function(ZONES, "needsMetadataOnlyFallback")

    assert "new MutationObserver" not in ZONES
    assert "consumeFrame(d);" in rendered
    assert "drawOverlay();" in rendered
    assert 'renderSpatialFrame(data, "rendered");' in ZONES
    assert 'renderSpatialFrame(fallbackData, "metadata-only");' in ZONES
    assert 'd.frame_transport === "metadata-only"' in fallback
    assert 'd.frame_transport === "binary-jpeg"' in fallback
    assert 'd.frame_transport === "base64-jpeg"' in fallback
    for correlation_field in ("job_id", "run_id", "frame_index"):
        assert correlation_field in frame_key
    assert "pendingSpatialFrame.key !== renderedKey" in ZONES
    assert "pendingSpatialFrame.key !== fallbackKey" in ZONES


def test_overlay_metric_is_local_correlated_and_exposed_through_debug_state():
    rendered = source_function(ZONES, "renderSpatialFrame")

    assert rendered.count("performance.now()") == 2
    assert "window.Perimetr.reportOverlayMetric" in rendered
    assert "overlay_ms: overlayMs" in rendered
    for correlation_field in (
        "camera_id",
        "job_id",
        "run_id",
        "frame_index",
        "frame_id",
        "timestamp",
    ):
        assert correlation_field in rendered
    assert "fetch(" not in rendered
    assert "window.Perimetr.getDebugState" in APP
    assert "overlayMs: 0" in APP
    assert "overlayAverageMs: 0" in APP
    assert "fpsLine.dataset.overlayMs" in APP


def test_runtime_switch_patches_are_serialized_and_failures_are_reconciled():
    flush = source_function(APP, "flushRuntimeProcessing")
    reconcile = source_function(APP, "reconcileRuntimeState")
    request = source_function(APP, "requestRuntimeState")

    assert "runtimeSyncSequence" not in APP
    assert "if (runtimeSyncRunning) return;" in flush
    assert "while (runtimeSyncCompletedRevision < runtimeSyncReadyRevision)" in flush
    assert "await requestRuntimeState(" in flush
    assert '"PATCH"' in flush
    assert "console.error(" in flush
    assert 'requestRuntimeState(context, "GET")' in reconcile
    assert "applyRuntimeOptions(payload.options);" in reconcile
    assert 'setRuntimeSyncStatus("error"' in reconcile
    assert 'requestOptions.body = JSON.stringify(options);' in request
    assert 'tag.setAttribute("aria-invalid", "true")' in APP
    assert 'tag.dataset.runtimeSync = status;' in APP


@pytest.mark.parametrize("asset", [APP_PATH, ZONES_PATH])
def test_changed_frontend_assets_are_valid_javascript(asset: Path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    result = subprocess.run(
        [node, "--check", str(asset)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
