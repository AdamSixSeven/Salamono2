import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
ZONES = (ROOT / "frontend" / "zones.js").read_text(encoding="utf-8")
INTERPOLATION_PATH = ROOT / "frontend" / "posture_interpolation.js"
INTERPOLATION = INTERPOLATION_PATH.read_text(encoding="utf-8")


def test_posture_interpolation_is_loaded_before_overlay_and_uses_short_raf():
    interpolation_asset = "posture_interpolation.js?v=2.3.0"
    overlay_asset = "zones.js?v=2.3.0"

    assert interpolation_asset in INDEX
    assert INDEX.index(interpolation_asset) < INDEX.index(overlay_asset)
    assert "durationMs: 45" in ZONES
    assert "window.requestAnimationFrame" in ZONES
    assert "window.cancelAnimationFrame" in ZONES
    assert "drawOverlay(timestamp)" in ZONES


def test_overlay_tracks_landmarks_per_track_and_resets_on_camera_change():
    assert "class TrackInterpolator" in INTERPOLATION
    assert "this.tracks = new Map()" in INTERPOLATION
    assert "previous: previous" in INTERPOLATION
    assert "current: cloneLandmarks(target)" in INTERPOLATION
    assert "postureInterpolator.sample(" in ZONES
    assert "postureInterpolator.update(" in ZONES

    camera_handler = ZONES.split(
        'document.addEventListener("perimetr-camera-changed"',
        1,
    )[1].split(
        'document.addEventListener("perimetr-layers"',
        1,
    )[0]
    assert "postureInterpolator.clear()" in camera_handler
    assert "cancelPostureAnimation()" in camera_handler


def test_interpolator_math_duplicate_frame_guard_and_private_copies():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    script = r"""
const assert = require("assert");
const api = require(process.argv[1]);
const pose = x => [[x, 0.2, 0.0, 0.9]];
const assessment = (timestamp, x, track = 7) => ({
    track_id: track,
    timestamp,
    pose_landmarks: pose(x),
});

const tracks = new api.TrackInterpolator({durationMs: 45});
const initial = assessment(1, 0);
tracks.update([initial], 0, "frame-1");
assert.strictEqual(tracks.size, 1);
assert.strictEqual(tracks.sample(7, 0)[0][0], 0);
initial.pose_landmarks[0][0] = 9;
assert.strictEqual(tracks.sample(7, 0)[0][0], 0);

// Large motion uses a short adaptive transition and ease-out behaviour.
const target = assessment(2, 1);
assert.strictEqual(tracks.update([target], 100, "frame-2"), true);
assert.strictEqual(tracks.isAnimating(109), true);
const halfway = tracks.sample(7, 109)[0][0];
assert(halfway > 0.5 && halfway < 1.0);

// Duplicate payload must not restart the transition.
tracks.update([target], 110, "frame-2");
assert.strictEqual(tracks.sample(7, 118)[0][0], 1);
assert.strictEqual(tracks.isAnimating(118), false);

tracks.update([assessment(3, 0.8)], 130, "frame-3");
const retargeted = tracks.sample(7, 139)[0][0];
assert(retargeted < 1.0 && retargeted > 0.8);

tracks.update([assessment(4, 0.4, 8)], 160, "frame-4");
assert.strictEqual(tracks.size, 1);
assert.strictEqual(tracks.sample(7, 160), null);
assert.strictEqual(tracks.sample(8, 160)[0][0], 0.4);
tracks.clear();
assert.strictEqual(tracks.size, 0);
"""
    result = subprocess.run(
        [node, "-e", script, os.fspath(INTERPOLATION_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize(
    "asset",
    [
        ROOT / "frontend" / "posture_interpolation.js",
        ROOT / "frontend" / "zones.js",
    ],
)
def test_posture_frontend_javascript_syntax(asset: Path):
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
