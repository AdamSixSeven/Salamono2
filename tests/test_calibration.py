import json
import os
import tempfile

import numpy as np
import pytest

from backend.calibration import (
    Calibration,
    CalibrationCompatibilityError,
    CalibrationStore,
    calibrate_from_markers,
)
from backend.marker_detector import MarkerDetection


def _synthetic_markers(marker_ids, image_corners):
    """Fake valid square markers whose centres are image_corners."""
    half_size = 5.0
    return [
        MarkerDetection(
            marker_id=mid,
            corners=[
                (center[0] - half_size, center[1] - half_size),
                (center[0] + half_size, center[1] - half_size),
                (center[0] + half_size, center[1] + half_size),
                (center[0] - half_size, center[1] + half_size),
            ],
        )
        for mid, center in zip(marker_ids, image_corners)
    ]


def test_calibrate_identity_grid():
    # Markers at pixel positions (0,0), (100,0), (100,100), (0,0+100). Reference
    # rectangle is 1m x 1m — one metre per 100 pixels.
    ids = [10, 20, 30, 40]
    image_corners = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    markers = _synthetic_markers(ids, image_corners)

    cal = calibrate_from_markers(
        camera_id="cam_default",
        detections=markers,
        marker_ids=ids,
        width_m=1.0,
        height_m=1.0,
    )
    # Middle of image should project to (0.5, 0.5) m.
    x, y = cal.project(50.0, 50.0)
    assert abs(x - 0.5) < 1e-6
    assert abs(y - 0.5) < 1e-6


def test_distance_and_inside():
    ids = [10, 20, 30, 40]
    image_corners = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    cal = calibrate_from_markers(
        camera_id="cam_default",
        detections=_synthetic_markers(ids, image_corners),
        marker_ids=ids,
        width_m=2.0,
        height_m=2.0,
    )
    # 100 px right of image → 2 m outside on X.
    d = cal.distance_to_boundary_m(200.0, 50.0)
    assert d == pytest.approx(2.0, abs=1e-6)
    assert cal.is_inside(200.0, 50.0) is False

    # Middle of image → inside, gap 1 m to nearest edge.
    d_inside = cal.distance_to_boundary_m(50.0, 50.0)
    assert d_inside == pytest.approx(-1.0, abs=1e-6)
    assert cal.is_inside(50.0, 50.0) is True


def test_calibrate_missing_marker_raises():
    ids = [10, 20, 30, 40]
    image_corners = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    # Only 3 markers detected — one required id missing.
    markers = _synthetic_markers(ids[:3], image_corners[:3])
    with pytest.raises(ValueError, match="Markers not detected"):
        calibrate_from_markers(
            camera_id="cam",
            detections=markers,
            marker_ids=ids,
            width_m=1.0,
            height_m=1.0,
        )


def test_calibrate_needs_positive_dimensions():
    ids = [10, 20, 30, 40]
    image_corners = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    markers = _synthetic_markers(ids, image_corners)
    with pytest.raises(ValueError, match="positive"):
        calibrate_from_markers(
            camera_id="cam",
            detections=markers,
            marker_ids=ids,
            width_m=0.0,
            height_m=1.0,
        )


def test_calibration_store_roundtrip(tmp_path):
    path = str(tmp_path / "cal.json")
    store = CalibrationStore(path)

    ids = [10, 20, 30, 40]
    image_corners = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    cal = calibrate_from_markers(
        camera_id="cam_a",
        detections=_synthetic_markers(ids, image_corners),
        marker_ids=ids,
        width_m=1.5,
        height_m=1.5,
    )
    store.set(cal)
    assert os.path.exists(path)

    # Reload from disk.
    store2 = CalibrationStore(path)
    got = store2.get("cam_a")
    assert got is not None
    assert got.marker_ids == ids
    assert got.width_m == 1.5
    # Homography should be preserved.
    x, y = got.project(50.0, 50.0)
    assert abs(x - 0.75) < 1e-6

    assert store2.clear("cam_a") is True
    assert store2.get("cam_a") is None


def test_normalized_homography_scales_across_matching_resolutions():
    ids = [10, 20, 30, 40]
    centers = [
        (100.0, 50.0),
        (900.0, 50.0),
        (900.0, 450.0),
        (100.0, 450.0),
    ]
    cal = calibrate_from_markers(
        camera_id="cam_scaled",
        detections=_synthetic_markers(ids, centers),
        marker_ids=ids,
        width_m=4.0,
        height_m=2.0,
        frame_width=1000,
        frame_height=500,
    )

    assert cal.source_frame_width == 1000
    assert cal.source_frame_height == 500
    assert cal.source_aspect_ratio == pytest.approx(2.0)
    assert cal.marker_centers[0] == pytest.approx([0.1, 0.1])
    assert cal.marker_centers_px[2] == pytest.approx([900.0, 450.0])
    assert cal.project(500, 250, 1000, 500) == pytest.approx((2.0, 1.0))
    assert cal.project(1000, 500, 2000, 1000) == pytest.approx((2.0, 1.0))
    assert cal.unproject(2.0, 1.0, 2000, 1000) == pytest.approx((1000, 500))


def test_aspect_compatibility_accepts_one_percent_and_rejects_more():
    ids = [10, 20, 30, 40]
    centers = [(100, 100), (900, 100), (900, 900), (100, 900)]
    cal = calibrate_from_markers(
        "cam_aspect",
        _synthetic_markers(ids, centers),
        ids,
        2.0,
        2.0,
        frame_width=1000,
        frame_height=1000,
    )

    assert cal.is_compatible(1010, 1000)
    assert cal.compatibility_warning(1010, 1000) is None
    assert not cal.is_compatible(1011, 1000)
    assert "aspect mismatch" in cal.compatibility_warning(1011, 1000).lower()
    with pytest.raises(CalibrationCompatibilityError, match="aspect mismatch"):
        cal.project(500, 500, 1011, 1000)


def test_quality_report_has_required_metrics_and_aliases():
    ids = [10, 20, 30, 40]
    centers = [(100, 100), (900, 100), (900, 900), (100, 900)]
    cal = calibrate_from_markers(
        "cam_quality",
        _synthetic_markers(ids, centers),
        ids,
        3.0,
        3.0,
        frame_width=1000,
        frame_height=1000,
    )
    quality = cal.quality
    assert quality["valid"] is True
    assert quality["status"] in {"good", "warning"}
    assert 0.0 <= quality["score"] <= 1.0
    assert quality["quadrilateral_area_ratio"] == quality["coverage_ratio"]
    assert quality["messages"] == quality["warnings"]
    assert quality["smallest_marker_area_px"] == pytest.approx(100.0)
    assert quality["reprojection_error_normalized"] < 1e-6


def test_rejects_small_or_malformed_marker_corners():
    ids = [10, 20, 30, 40]
    valid = _synthetic_markers(
        ids, [(100, 100), (900, 100), (900, 900), (100, 900)]
    )
    valid[0] = MarkerDetection(
        marker_id=10,
        corners=[(99, 99), (101, 99), (101, 101), (99, 101)],
    )
    with pytest.raises(ValueError, match="too small"):
        calibrate_from_markers(
            "cam_small",
            valid,
            ids,
            2,
            2,
            frame_width=1000,
            frame_height=1000,
        )

    malformed = _synthetic_markers(
        ids, [(100, 100), (900, 100), (900, 900), (100, 900)]
    )
    malformed[1] = MarkerDetection(
        marker_id=20,
        corners=[(1, 1), (2, 1), (2, float("nan")), (1, 2)],
    )
    with pytest.raises(ValueError, match="4 finite corners"):
        calibrate_from_markers(
            "cam_nan",
            malformed,
            ids,
            2,
            2,
            frame_width=1000,
            frame_height=1000,
        )


def test_rejects_swapped_or_tiny_reference_geometry():
    ids = [10, 20, 30, 40]
    swapped = [(100, 100), (100, 900), (900, 900), (900, 100)]
    with pytest.raises(ValueError, match="order|winding|semantics"):
        calibrate_from_markers(
            "cam_swapped",
            _synthetic_markers(ids, swapped),
            ids,
            2,
            2,
            frame_width=1000,
            frame_height=1000,
        )

    tiny = [(100, 100), (140, 100), (140, 140), (100, 140)]
    with pytest.raises(ValueError, match="too small|too short"):
        calibrate_from_markers(
            "cam_tiny",
            _synthetic_markers(ids, tiny),
            ids,
            2,
            2,
            frame_width=1000,
            frame_height=1000,
        )


def test_store_migrates_legacy_record_without_losing_other_cameras(tmp_path):
    path = tmp_path / "legacy-calibration.json"
    path.write_text(
        json.dumps({
            "legacy_cam": {
                "camera_id": "legacy_cam",
                "marker_ids": [10, 20, 30, 40],
                "width_m": 1.0,
                "height_m": 1.0,
                "homography": [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]],
                "created_at": 123.0,
            },
            "broken_cam": {
                "camera_id": "broken_cam",
                "marker_ids": [10, 20, 30, 40],
                "width_m": 1.0,
                "height_m": 1.0,
                "homography": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            },
            "old_aspect_name": {
                "camera_id": "old_aspect_name",
                "marker_ids": [10, 20, 30, 40],
                "width_m": 1.0,
                "height_m": 1.0,
                "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "source_frame_width": 640,
                "source_frame_height": 480,
                "source_frame_aspect": 1.333333,
                "marker_centers": [[0, 0], [1, 0], [1, 1], [0, 1]],
            },
        }),
        encoding="utf-8",
    )

    store = CalibrationStore(str(path))
    legacy = store.get("legacy_cam")
    assert legacy is not None
    assert legacy.project(50, 50) == pytest.approx((0.5, 0.5))
    assert legacy.is_compatible(100, 100) is False
    assert "recalibrate" in legacy.compatibility_warning(100, 100).lower()
    assert legacy.quality["status"] == "invalid"
    assert store.get("broken_cam") is None
    renamed = store.get("old_aspect_name")
    assert renamed is not None
    assert renamed.source_aspect_ratio == pytest.approx(4 / 3)
    assert renamed.is_compatible(1280, 960)


def test_store_replaces_file_atomically_and_rolls_back_failed_update(
    tmp_path, monkeypatch
):
    path = tmp_path / "calibration.json"
    ids = [10, 20, 30, 40]
    points = [(10, 10), (90, 10), (90, 90), (10, 90)]
    store = CalibrationStore(str(path))
    first = calibrate_from_markers(
        "cam_atomic",
        _synthetic_markers(ids, points),
        ids,
        2.0,
        1.0,
        frame_width=100,
        frame_height=100,
    )
    store.set(first)
    persisted_before = path.read_text(encoding="utf-8")

    replacement = calibrate_from_markers(
        "cam_atomic",
        _synthetic_markers(ids, points),
        ids,
        4.0,
        2.0,
        frame_width=100,
        frame_height=100,
    )

    def fail_replace(_source, _target):
        raise OSError("simulated interrupted replace")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        store.set(replacement)

    assert path.read_text(encoding="utf-8") == persisted_before
    assert not path.with_suffix(".json.tmp").exists()
    assert store.get("cam_atomic").width_m == pytest.approx(2.0)
