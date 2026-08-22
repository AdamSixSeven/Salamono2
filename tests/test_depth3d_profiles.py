from __future__ import annotations

import numpy as np

from backend.depth3d import Depth3DCalibration
from backend.depth3d_checkerboard import CheckerboardObservation, CheckerboardSpec
from backend.depth3d_profiles import Depth3DProfileStore


def _observation(camera_id: str, timestamp: float, offset: float = 0.0) -> CheckerboardObservation:
    corners = []
    for row in range(6):
        for col in range(9):
            corners.append([100 + col * 35 + offset, 80 + row * 35 + offset * 0.3])
    return CheckerboardObservation(
        camera_id=camera_id,
        timestamp=timestamp,
        frame_width=960,
        frame_height=720,
        image_corners=np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
        coverage_ratio=0.24,
        blur_score=180.0,
    )


def _calibration(camera_id: str) -> Depth3DCalibration:
    return Depth3DCalibration(
        camera_id=camera_id,
        source_frame_width=960,
        source_frame_height=720,
        camera_matrix=[[800.0, 0.0, 480.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]],
        dist_coeffs=[0.1, -0.02, 0.0, 0.0, 0.0],
        rms_error_px=0.31,
        mean_reprojection_error_px=0.28,
        views_used=1,
        board_spec={
            "inner_corners_x": 9,
            "inner_corners_y": 6,
            "square_length_m": 0.03,
            "lens_model": "pinhole",
            "pattern_type": "checkerboard",
            "per_view_errors_px": [0.28],
        },
    )


def test_profiles_and_views_survive_restart(tmp_path):
    root = tmp_path / "profiles"
    store = Depth3DProfileStore(str(root))
    profile = store.create_profile("cam_phone_1", "Telefon 1x", CheckerboardSpec())
    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    accepted, warning, view = store.add_view(
        "cam_phone_1", _observation("cam_phone_1", 1.0), frame, CheckerboardSpec(), profile["profile_id"]
    )
    assert accepted is True
    assert warning is None
    store.save_calibration("cam_phone_1", profile["profile_id"], _calibration("cam_phone_1"))

    reloaded = Depth3DProfileStore(str(root))
    assert reloaded.active_profile_id("cam_phone_1") == profile["profile_id"]
    assert len(reloaded.list_views("cam_phone_1", profile["profile_id"])) == 1
    assert reloaded.get_calibration("cam_phone_1").mean_reprojection_error_px == 0.28
    assert reloaded.get_view("cam_phone_1", profile["profile_id"], view["view_id"])["reprojection_error_px"] == 0.28


def test_deleting_view_invalidates_calibration(tmp_path):
    store = Depth3DProfileStore(str(tmp_path / "profiles"))
    profile = store.create_profile("cam", "Main", CheckerboardSpec())
    accepted, _, view = store.add_view(
        "cam", _observation("cam", 1.0), np.zeros((720, 960, 3), np.uint8), CheckerboardSpec(), profile["profile_id"]
    )
    assert accepted
    store.save_calibration("cam", profile["profile_id"], _calibration("cam"))
    assert store.get_calibration("cam") is not None

    store.delete_view("cam", profile["profile_id"], view["view_id"])
    assert store.get_calibration("cam") is None
    assert store.list_views("cam", profile["profile_id"]) == []


def test_profiles_are_separate_and_can_be_activated(tmp_path):
    store = Depth3DProfileStore(str(tmp_path / "profiles"))
    first = store.create_profile("cam", "1x", CheckerboardSpec(), activate=True)
    second = store.create_profile("cam", "fisheye", CheckerboardSpec(lens_model="fisheye"), activate=True)
    assert store.active_profile_id("cam") == second["profile_id"]
    store.activate("cam", first["profile_id"])
    assert store.active_profile_id("cam") == first["profile_id"]
    assert len(store.list_profiles("cam")) == 2
