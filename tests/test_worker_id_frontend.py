import os
import re
import shutil
import subprocess
from pathlib import Path

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


def numeric_constant(source: str, name: str) -> int:
    match = re.search(rf"const {re.escape(name)} = (\d+);", source)
    assert match is not None
    return int(match.group(1))


def test_worker_identity_state_is_keyed_by_track_with_worker_fallback():
    app_key = source_function(APP, "workerUiTrackKey")
    overlay_key = source_function(ZONES, "workerOverlayTrackKey")

    for source in (app_key, overlay_key):
        assert "identity.track_id !== null" in source
        assert "identity.track_id !== undefined" in source
        assert '"track:" + String(identity.track_id)' in source
        assert '"worker:" + String(identity && identity.worker_id || "")' in source

    stable = source_function(APP, "stableWorkerIdentities")
    assert "const key = workerUiTrackKey(identity)" in stable
    assert "previous.identity.worker_id === identity.worker_id" in stable
    assert "identity," in stable


def test_camera_change_clears_worker_profile_and_spatial_tracks():
    select_camera = source_function(APP, "selectCamera")
    assert "workerUiTracks.clear()" in select_camera

    camera_handler = ZONES.split(
        'document.addEventListener("perimetr-camera-changed"',
        1,
    )[1].split(
        'document.addEventListener("perimetr-layers"',
        1,
    )[0]
    assert "clearWorkerOverlayTracks()" in camera_handler
    clear = source_function(ZONES, "clearWorkerOverlayTracks")
    assert "workerOverlayTracks.clear()" in clear
    assert "workerIdentifications = []" in clear


def test_profile_label_uses_current_track_bbox_and_does_not_merge_workers():
    update = source_function(ZONES, "updateWorkerOverlayTracks")
    snapshot = source_function(ZONES, "workerOverlaySnapshot")
    draw = source_function(ZONES, "drawWorkerIds")

    assert "existing.identity.worker_id === identity.worker_id" in update
    assert "const targetBox = identity.person_box.map(Number)" in update
    assert "targetBox: targetBox" in update
    assert "person_box: sampled.box" in snapshot
    assert "const box = identity.person_box" in draw
    assert "identity.full_name" in draw


def test_marker_corners_are_current_frame_only():
    polygon = source_function(ZONES, "currentWorkerTagPolygon")
    update = source_function(ZONES, "updateWorkerOverlayTracks")
    snapshot = source_function(ZONES, "workerOverlaySnapshot")

    assert "identity.cached === true" in polygon
    assert "return []" in polygon
    assert "const seenKeys = new Set()" in update
    assert "seenKeys.add(key)" in update
    assert "if (!seenKeys.has(key))" in update
    assert "track.startPolygon = []" in update
    assert "track.targetPolygon = []" in update
    assert "tag_polygon: sampled.polygon" in snapshot


def test_missing_worker_bbox_has_only_a_short_fade():
    hold = numeric_constant(ZONES, "WORKER_OVERLAY_HOLD_MS")
    fade = numeric_constant(ZONES, "WORKER_OVERLAY_FADE_MS")

    assert hold + fade <= 500
    snapshot = source_function(ZONES, "workerOverlaySnapshot")
    assert "age > maxAge" in snapshot
    assert "workerOverlayTracks.delete(key)" in snapshot


def test_track_key_and_marker_freshness_helpers_execute_in_node():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    functions = "\n".join([
        source_function(ZONES, "validPolygon"),
        source_function(ZONES, "workerOverlayTrackKey"),
        source_function(ZONES, "currentWorkerTagPolygon"),
    ])
    script = functions + r"""
const assert = require("assert");
assert.strictEqual(
    workerOverlayTrackKey({worker_id: "W-001", track_id: 7}),
    "track:7"
);
assert.strictEqual(
    workerOverlayTrackKey({worker_id: "W-001"}),
    "worker:W-001"
);
const polygon = [[1, 2], [3, 4], [5, 6]];
assert.deepStrictEqual(
    currentWorkerTagPolygon({cached: false, tag_polygon: polygon}),
    polygon
);
assert.deepStrictEqual(
    currentWorkerTagPolygon({cached: true, tag_polygon: polygon}),
    []
);
assert.deepStrictEqual(
    currentWorkerTagPolygon({cached: false, tag_polygon: []}),
    []
);
"""
    result = subprocess.run(
        [node, "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize("asset", [APP_PATH, ZONES_PATH])
def test_worker_id_frontend_javascript_syntax(asset: Path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    result = subprocess.run(
        [node, "--check", os.fspath(asset)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
