from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.depth3d import (
    Depth3DCalibration,
    Depth3DCalibrationStore,
    Depth3DUnavailableError,
    MonocularDepthEstimator,
    depth_to_xyz,
    estimate_person_depth,
    pixel_to_point,
)
from backend.depth3d_checkerboard import (
    CheckerboardObservation,
    CheckerboardSpec,
    calibrate_checkerboard_intrinsics,
    checkerboard_object_points,
    detect_checkerboard,
    render_checkerboard_png,
)


def test_checkerboard_png_is_detectable():
    spec = CheckerboardSpec()
    payload = render_checkerboard_png(spec, dpi=120)
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    corners, diagnostics = detect_checkerboard(image, spec)

    assert payload.startswith(b"\x89PNG")
    assert corners is not None
    assert len(corners) == spec.expected_corners
    assert diagnostics["coverage_ratio"] > 0.3


def test_depth_to_xyz_uses_pinhole_geometry():
    depth = np.full((3, 3), 2.0, dtype=np.float32)
    matrix = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
    points, valid = depth_to_xyz(depth, matrix)
    assert valid.all()
    assert np.allclose(points[4], [0.0, 0.0, 2.0])
    assert np.allclose(points[0], [-1.0, -1.0, 2.0])


def test_pixel_to_point_uses_local_depth_median():
    depth = np.full((9, 9), 4.0, dtype=np.float32)
    depth[4, 4] = np.nan
    matrix = np.array([[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]])
    point = pixel_to_point(depth, matrix, (4, 4), radius=1)
    assert np.allclose(point, [0.0, 0.0, 4.0])


def test_intrinsic_calibration_from_synthetic_checkerboard_views():
    spec = CheckerboardSpec(inner_corners_x=9, inner_corners_y=6, square_length_m=0.03)
    object_points = checkerboard_object_points(spec)
    width, height = 960, 720
    true_matrix = np.array([[780.0, 0.0, width / 2], [0.0, 785.0, height / 2], [0.0, 0.0, 1.0]], dtype=np.float64)
    observations = []
    for index in range(14):
        rvec = np.array([0.03 + index * 0.013, -0.19 + index * 0.025, -0.10 + index * 0.017])
        tvec = np.array([-0.13 + index * 0.019, -0.08 + (index % 4) * 0.045, 0.65 + index * 0.045])
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, true_matrix, np.zeros(5))
        observations.append(CheckerboardObservation(
            camera_id="cam-test",
            timestamp=float(index),
            frame_width=width,
            frame_height=height,
            image_corners=projected.astype(np.float32),
            coverage_ratio=0.15,
            blur_score=200.0,
        ))

    calibration = calibrate_checkerboard_intrinsics("cam-test", observations, spec)
    matrix = np.asarray(calibration.camera_matrix)
    assert calibration.views_used == 14
    assert calibration.mean_reprojection_error_px < 0.1
    assert abs(matrix[0, 0] - true_matrix[0, 0]) < 5
    assert abs(matrix[1, 1] - true_matrix[1, 1]) < 5
    assert calibration.board_spec["pattern_type"] == "checkerboard"


def test_calibration_store_round_trip(tmp_path: Path):
    path = tmp_path / "depth3d.json"
    store = Depth3DCalibrationStore(str(path))
    calibration = Depth3DCalibration(
        camera_id="cam-a",
        source_frame_width=960,
        source_frame_height=720,
        camera_matrix=[[800, 0, 480], [0, 800, 360], [0, 0, 1]],
        dist_coeffs=[0, 0, 0, 0, 0],
        rms_error_px=0.2,
        mean_reprojection_error_px=0.15,
        views_used=12,
        board_spec={"pattern_type": "checkerboard", "lens_model": "pinhole"},
    )
    store.put(calibration)
    reloaded = Depth3DCalibrationStore(str(path)).get("cam-a")
    assert reloaded is not None
    assert reloaded.camera_matrix_for(1920, 1440)[0, 0] == pytest.approx(1600)
    assert reloaded.compatibility_warning(1280, 720) is not None


def test_disabled_model_fails_without_importing_transformers():
    estimator = MonocularDepthEstimator(enabled=False, model_id="unused", device="cpu")
    with pytest.raises(Depth3DUnavailableError):
        estimator.infer(np.zeros((32, 32, 3), dtype=np.uint8))


def test_person_depth_uses_robust_torso_median_and_ray_geometry():
    depth = np.full((100, 100), 8.0, dtype=np.float32)
    depth[20:75, 30:70] = 3.0
    depth[30, 45] = 40.0
    matrix = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    estimate = estimate_person_depth(
        depth, matrix, (20, 10, 80, 90), confidence=0.9, min_depth_m=0.2, max_depth_m=20.0
    )
    assert estimate is not None
    assert estimate.depth_z_m == pytest.approx(3.0)
    assert estimate.ray_distance_m >= estimate.depth_z_m
    assert estimate.valid_ratio > 0.9
    assert estimate.sample_count >= 20
