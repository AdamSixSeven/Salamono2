"""Traditional checkerboard calibration for the monocular depth preview."""
from __future__ import annotations

import io
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from backend.depth3d import Depth3DCalibration, IntrinsicCalibrationError

MIN_CALIBRATION_VIEWS = 8


@dataclass(frozen=True)
class CheckerboardSpec:
    """Checkerboard dimensions use the number of internal black/white intersections."""

    inner_corners_x: int = 9
    inner_corners_y: int = 6
    square_length_m: float = 0.03
    lens_model: str = "pinhole"

    def validate(self) -> "CheckerboardSpec":
        if not 3 <= self.inner_corners_x <= 20 or not 3 <= self.inner_corners_y <= 20:
            raise IntrinsicCalibrationError("Szachownica wymaga od 3 do 20 narożników w każdym kierunku.")
        if not math.isfinite(self.square_length_m) or self.square_length_m <= 0:
            raise IntrinsicCalibrationError("Rozmiar pola musi być dodatni.")
        if self.lens_model not in {"pinhole", "fisheye"}:
            raise IntrinsicCalibrationError("Model obiektywu musi mieć wartość pinhole albo fisheye.")
        return self

    @property
    def pattern_size(self) -> tuple[int, int]:
        return int(self.inner_corners_x), int(self.inner_corners_y)

    @property
    def squares_x(self) -> int:
        return self.inner_corners_x + 1

    @property
    def squares_y(self) -> int:
        return self.inner_corners_y + 1

    @property
    def expected_corners(self) -> int:
        return self.inner_corners_x * self.inner_corners_y


@dataclass(frozen=True)
class CheckerboardObservation:
    camera_id: str
    timestamp: float
    frame_width: int
    frame_height: int
    image_corners: np.ndarray
    coverage_ratio: float
    blur_score: float

    @property
    def corner_count(self) -> int:
        return int(len(self.image_corners))

    @property
    def marker_count(self) -> int:
        return 0

    def pose_signature(self) -> tuple[float, float, float, float]:
        points = self.image_corners.reshape(-1, 2).astype(np.float64)
        width = max(1.0, float(self.frame_width))
        height = max(1.0, float(self.frame_height))
        center = points.mean(axis=0)
        span = np.ptp(points, axis=0)
        top = points[: max(2, min(len(points), 6))]
        vector = top[-1] - top[0]
        return (
            float(center[0] / width),
            float(center[1] / height),
            float(math.sqrt(max(0.0, span[0] * span[1] / (width * height)))),
            float(math.atan2(float(vector[1]), float(vector[0]))),
        )


def render_checkerboard_png(spec: CheckerboardSpec, *, dpi: int = 300, margin_mm: float = 10.0) -> bytes:
    spec.validate()
    if not 72 <= dpi <= 1200:
        raise IntrinsicCalibrationError("DPI musi mieścić się w zakresie 72–1200.")
    px_per_m = dpi / 0.0254
    square_px = max(24, int(round(spec.square_length_m * px_per_m)))
    board = np.full((spec.squares_y * square_px, spec.squares_x * square_px), 255, np.uint8)
    for row in range(spec.squares_y):
        for col in range(spec.squares_x):
            if (row + col) % 2 == 0:
                y1, y2 = row * square_px, (row + 1) * square_px
                x1, x2 = col * square_px, (col + 1) * square_px
                board[y1:y2, x1:x2] = 0
    margin_px = max(0, int(round((margin_mm / 1000.0) * px_per_m)))
    if margin_px:
        board = cv2.copyMakeBorder(board, margin_px, margin_px, margin_px, margin_px, cv2.BORDER_CONSTANT, value=255)
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise IntrinsicCalibrationError("Pillow jest wymagany do generowania PNG.") from exc
    output = io.BytesIO()
    Image.fromarray(board).save(output, format="PNG", dpi=(dpi, dpi))
    return output.getvalue()


def _find_sb(gray: np.ndarray, pattern_size: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
    return bool(found), corners


def detect_checkerboard(frame_bgr: np.ndarray, spec: CheckerboardSpec) -> tuple[np.ndarray | None, dict[str, Any]]:
    spec.validate()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    variants = [("gray", gray)]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(("clahe", clahe.apply(gray)))
    corners = None
    method = "none"
    for method_name, image in variants:
        found, candidate = _find_sb(image, spec.pattern_size)
        if found and candidate is not None:
            corners, method = candidate.astype(np.float32), method_name
            break
    if corners is None:
        scale = 1.5 if max(gray.shape[:2]) < 1400 else 1.25
        enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        found, candidate = _find_sb(enlarged, spec.pattern_size)
        if found and candidate is not None:
            corners = (candidate.astype(np.float32) / scale)
            method = "upscaled"
    height, width = gray.shape
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    coverage = 0.0
    if corners is not None:
        points = corners.reshape(-1, 2)
        span = np.ptp(points, axis=0)
        coverage = float(max(0.0, span[0] * span[1] / max(1.0, width * height)))
    return corners, {
        "found": corners is not None,
        "method": method,
        "corner_count": 0 if corners is None else int(len(corners)),
        "expected_corners": spec.expected_corners,
        "coverage_ratio": coverage,
        "blur_score": blur_score,
    }


def make_checkerboard_observation(camera_id: str, timestamp: float, frame_bgr: np.ndarray, spec: CheckerboardSpec) -> CheckerboardObservation:
    corners, diagnostics = detect_checkerboard(frame_bgr, spec)
    if corners is None or len(corners) != spec.expected_corners:
        raise IntrinsicCalibrationError(
            f"Nie wykryto pełnej szachownicy {spec.inner_corners_x}×{spec.inner_corners_y}. "
            f"Wykryto {diagnostics['corner_count']} z {spec.expected_corners} narożników."
        )
    if diagnostics["blur_score"] < 35:
        raise IntrinsicCalibrationError("Obraz szachownicy jest zbyt rozmyty. Unieruchom telefon i spróbuj ponownie.")
    if diagnostics["coverage_ratio"] < 0.025:
        raise IntrinsicCalibrationError("Szachownica jest zbyt mała w obrazie. Zbliż ją do kamery.")
    height, width = frame_bgr.shape[:2]
    return CheckerboardObservation(
        camera_id=camera_id,
        timestamp=float(timestamp),
        frame_width=int(width),
        frame_height=int(height),
        image_corners=corners.copy(),
        coverage_ratio=float(diagnostics["coverage_ratio"]),
        blur_score=float(diagnostics["blur_score"]),
    )


def checkerboard_object_points(spec: CheckerboardSpec) -> np.ndarray:
    points = np.zeros((spec.expected_corners, 3), np.float32)
    points[:, :2] = np.mgrid[0:spec.inner_corners_x, 0:spec.inner_corners_y].T.reshape(-1, 2)
    points[:, :2] *= float(spec.square_length_m)
    return points


def calibrate_checkerboard_intrinsics(camera_id: str, observations: list[CheckerboardObservation], spec: CheckerboardSpec) -> Depth3DCalibration:
    spec.validate()
    if len(observations) < MIN_CALIBRATION_VIEWS:
        raise IntrinsicCalibrationError(f"Zapisz co najmniej {MIN_CALIBRATION_VIEWS} różnych widoków szachownicy.")
    width, height = observations[0].frame_width, observations[0].frame_height
    if any((item.frame_width, item.frame_height) != (width, height) for item in observations):
        raise IntrinsicCalibrationError("Wszystkie widoki kalibracyjne muszą mieć tę samą rozdzielczość i orientację.")

    obj = checkerboard_object_points(spec)
    object_points = [obj.copy() for _ in observations]
    image_points = [item.image_corners.reshape(-1, 2).astype(np.float32) for item in observations]

    if spec.lens_model == "fisheye":
        K = np.array([[0.8 * width, 0, width / 2], [0, 0.8 * width, height / 2], [0, 0, 1]], np.float64)
        D = np.zeros((4, 1), np.float64)
        obj_f = [points.reshape(-1, 1, 3).astype(np.float64) for points in object_points]
        img_f = [points.reshape(-1, 1, 2).astype(np.float64) for points in image_points]
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-7)
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(obj_f, img_f, (width, height), K, D, flags=flags, criteria=criteria)
        projections = [cv2.fisheye.projectPoints(points, r, t, K, D)[0].reshape(-1, 2) for points, r, t in zip(obj_f, rvecs, tvecs)]
        dist_coeffs = D
    else:
        rms, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (width, height), None, None)
        projections = [cv2.projectPoints(points, r, t, K, dist_coeffs)[0].reshape(-1, 2) for points, r, t in zip(object_points, rvecs, tvecs)]

    if not math.isfinite(float(rms)) or not np.all(np.isfinite(K)):
        raise IntrinsicCalibrationError("Kalibracja zwróciła nieprawidłowe parametry.")
    errors = [float(np.mean(np.linalg.norm(projected - measured, axis=1))) for projected, measured in zip(projections, image_points)]
    mean_error = float(np.mean(errors))
    if mean_error > 2.0:
        raise IntrinsicCalibrationError(f"Błąd reprojekcji {mean_error:.2f}px jest zbyt duży. Zbierz ostrzejsze i bardziej zróżnicowane widoki.")

    all_points = np.concatenate(image_points, axis=0)
    span = np.ptp(all_points, axis=0)
    aggregate_coverage = float(span[0] * span[1] / max(1.0, width * height))
    centers = np.asarray([points.mean(axis=0) for points in image_points])
    center_span = np.ptp(centers, axis=0) / np.asarray([width, height], dtype=np.float64)
    quality = "good" if mean_error <= 0.8 and aggregate_coverage >= 0.55 else "warning"

    board_spec = asdict(spec)
    board_spec.update({
        "pattern_type": "checkerboard",
        "aggregate_coverage_ratio": round(aggregate_coverage, 6),
        "center_span_x": round(float(center_span[0]), 6),
        "center_span_y": round(float(center_span[1]), 6),
        "quality": quality,
        "per_view_errors_px": [round(float(error), 6) for error in errors],
    })
    return Depth3DCalibration(
        camera_id=camera_id,
        source_frame_width=width,
        source_frame_height=height,
        camera_matrix=K.tolist(),
        dist_coeffs=np.asarray(dist_coeffs).reshape(-1).tolist(),
        rms_error_px=float(rms),
        mean_reprojection_error_px=mean_error,
        views_used=len(observations),
        board_spec=board_spec,
        depth_scale=1.0,
        created_at=time.time(),
        updated_at=time.time(),
    )


def undistort_with_calibration(frame_bgr: np.ndarray, calibration: Depth3DCalibration) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame_bgr.shape[:2]
    K = calibration.camera_matrix_for(width, height)
    D = calibration.dist_array()
    lens_model = str(calibration.board_spec.get("lens_model", "pinhole"))
    if lens_model == "fisheye":
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D[:4], (width, height), np.eye(3), balance=0.0)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D[:4], np.eye(3), new_K, (width, height), cv2.CV_16SC2)
        return cv2.remap(frame_bgr, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), new_K
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (width, height), alpha=0.0, newImgSize=(width, height))
    return cv2.undistort(frame_bgr, K, D, None, new_K), new_K


def approximate_camera_matrix(width: int, height: int, horizontal_fov_deg: float = 65.0) -> np.ndarray:
    focal = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    return np.asarray([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]], dtype=np.float64)
