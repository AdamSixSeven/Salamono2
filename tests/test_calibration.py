import os
import tempfile

import numpy as np
import pytest

from backend.calibration import (
    Calibration,
    CalibrationStore,
    calibrate_from_markers,
)
from backend.marker_detector import MarkerDetection


def _synthetic_markers(marker_ids, image_corners):
    """Fake markers whose centres are the requested image_corners.

    Each corner is used as the marker centre (all 4 corner-points collapse
    to the same point — good enough for the calibration math, which only
    uses the centre).
    """
    return [
        MarkerDetection(marker_id=mid, corners=[c, c, c, c])
        for mid, c in zip(marker_ids, image_corners)
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
