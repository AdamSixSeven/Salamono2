"""Monocular metric-depth support, intrinsic calibration and point-cloud geometry.

The module deliberately treats single-camera depth as an estimate.  Camera
intrinsics are calibrated from multiple ChArUco observations; a metric-depth
network supplies per-pixel Z values; and calibrated pinhole geometry converts
those values to XYZ coordinates in the camera frame.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np


DEFAULT_DICTIONARY_NAME = "DICT_4X4_50"
ASPECT_TOLERANCE = 0.01
MIN_CAPTURE_CORNERS = 10
MIN_CALIBRATION_VIEWS = 8


class Depth3DError(RuntimeError):
    """Base class for actionable 3D-module failures."""


class Depth3DUnavailableError(Depth3DError):
    """The optional neural model cannot be used in this installation."""


class IntrinsicCalibrationError(Depth3DError):
    """ChArUco observations cannot produce a trustworthy camera model."""


@dataclass(frozen=True)
class CharucoBoardSpec:
    squares_x: int = 7
    squares_y: int = 5
    square_length_m: float = 0.04
    marker_length_m: float = 0.03
    dictionary_name: str = DEFAULT_DICTIONARY_NAME

    def validate(self) -> "CharucoBoardSpec":
        if self.squares_x < 3 or self.squares_y < 3:
            raise IntrinsicCalibrationError("ChArUco board needs at least 3x3 squares.")
        if not math.isfinite(self.square_length_m) or self.square_length_m <= 0:
            raise IntrinsicCalibrationError("square_length_m must be positive.")
        if not math.isfinite(self.marker_length_m) or self.marker_length_m <= 0:
            raise IntrinsicCalibrationError("marker_length_m must be positive.")
        if self.marker_length_m >= self.square_length_m:
            raise IntrinsicCalibrationError(
                "marker_length_m must be smaller than square_length_m."
            )
        if not hasattr(cv2.aruco, self.dictionary_name):
            raise IntrinsicCalibrationError(
                f"Unsupported ArUco dictionary: {self.dictionary_name}"
            )
        return self

    @property
    def board_width_m(self) -> float:
        return self.squares_x * self.square_length_m

    @property
    def board_height_m(self) -> float:
        return self.squares_y * self.square_length_m


def create_charuco_board(spec: CharucoBoardSpec):
    spec.validate()
    dictionary_id = getattr(cv2.aruco, spec.dictionary_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.CharucoBoard(
        (int(spec.squares_x), int(spec.squares_y)),
        float(spec.square_length_m),
        float(spec.marker_length_m),
        dictionary,
    )


def render_charuco_png(
    spec: CharucoBoardSpec,
    *,
    dpi: int = 300,
    margin_mm: float = 10.0,
) -> bytes:
    """Render a print-scale ChArUco PNG with DPI metadata."""
    spec.validate()
    if dpi < 72 or dpi > 1200:
        raise IntrinsicCalibrationError("dpi must be between 72 and 1200.")
    if margin_mm < 0 or margin_mm > 100:
        raise IntrinsicCalibrationError("margin_mm must be between 0 and 100.")

    px_per_m = dpi / 0.0254
    board_w = max(320, int(round(spec.board_width_m * px_per_m)))
    board_h = max(240, int(round(spec.board_height_m * px_per_m)))
    margin_px = int(round((margin_mm / 1000.0) * px_per_m))
    board = create_charuco_board(spec)
    image = board.generateImage((board_w, board_h), marginSize=0, borderBits=1)
    if margin_px:
        image = cv2.copyMakeBorder(
            image,
            margin_px,
            margin_px,
            margin_px,
            margin_px,
            cv2.BORDER_CONSTANT,
            value=255,
        )

    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - Pillow is already a qrcode dep
        raise Depth3DUnavailableError("Pillow is required to generate the board PNG.") from exc

    output = io.BytesIO()
    Image.fromarray(image).save(output, format="PNG", dpi=(dpi, dpi))
    return output.getvalue()


@dataclass(frozen=True)
class CharucoObservation:
    camera_id: str
    timestamp: float
    frame_width: int
    frame_height: int
    charuco_corners: np.ndarray
    charuco_ids: np.ndarray
    marker_count: int

    @property
    def corner_count(self) -> int:
        return int(len(self.charuco_ids))

    def pose_signature(self) -> tuple[float, float, float, float]:
        points = self.charuco_corners.reshape(-1, 2).astype(np.float64)
        width = max(1.0, float(self.frame_width))
        height = max(1.0, float(self.frame_height))
        center = points.mean(axis=0)
        span = np.ptp(points, axis=0)
        if len(points) >= 2:
            vector = points[-1] - points[0]
            angle = math.atan2(float(vector[1]), float(vector[0]))
        else:
            angle = 0.0
        return (
            float(center[0] / width),
            float(center[1] / height),
            float(math.sqrt(max(0.0, span[0] * span[1] / (width * height)))),
            float(angle),
        )


def detect_charuco(
    frame_bgr: np.ndarray,
    spec: CharucoBoardSpec,
    *,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, list[np.ndarray], np.ndarray | None]:
    board = create_charuco_board(spec)
    params = cv2.aruco.CharucoParameters()
    if camera_matrix is not None:
        params.cameraMatrix = np.asarray(camera_matrix, dtype=np.float64)
    if dist_coeffs is not None:
        params.distCoeffs = np.asarray(dist_coeffs, dtype=np.float64)
    detector = cv2.aruco.CharucoDetector(board, params)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
    return corners, ids, marker_corners, marker_ids


class IntrinsicCaptureStore:
    """Bounded, in-memory ChArUco observations per camera."""

    def __init__(self, max_cameras: int = 16, max_views_per_camera: int = 40):
        self.max_cameras = max_cameras
        self.max_views_per_camera = max_views_per_camera
        self._lock = threading.Lock()
        self._items: OrderedDict[str, list[CharucoObservation]] = OrderedDict()

    def add(self, observation: CharucoObservation) -> tuple[bool, str | None]:
        signature = observation.pose_signature()
        with self._lock:
            views = self._items.setdefault(observation.camera_id, [])
            for existing in views:
                old = existing.pose_signature()
                center_distance = math.hypot(signature[0] - old[0], signature[1] - old[1])
                scale_distance = abs(signature[2] - old[2])
                angle_distance = abs(math.atan2(math.sin(signature[3] - old[3]), math.cos(signature[3] - old[3])))
                if center_distance < 0.035 and scale_distance < 0.025 and angle_distance < math.radians(6):
                    return False, "Widok jest zbyt podobny do już zapisanej klatki. Zmień pozycję lub kąt planszy."
            views.append(observation)
            if len(views) > self.max_views_per_camera:
                del views[0 : len(views) - self.max_views_per_camera]
            self._items.move_to_end(observation.camera_id)
            while len(self._items) > self.max_cameras:
                self._items.popitem(last=False)
        return True, None

    def get(self, camera_id: str) -> list[CharucoObservation]:
        with self._lock:
            return list(self._items.get(camera_id, []))

    def clear(self, camera_id: str) -> None:
        with self._lock:
            self._items.pop(camera_id, None)


@dataclass
class Depth3DCalibration:
    camera_id: str
    source_frame_width: int
    source_frame_height: int
    camera_matrix: list[list[float]]
    dist_coeffs: list[float]
    rms_error_px: float
    mean_reprojection_error_px: float
    views_used: int
    board_spec: dict[str, Any]
    depth_scale: float = 1.0
    depth_scale_median_error_m: float | None = None
    world_from_camera: list[list[float]] | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def source_aspect_ratio(self) -> float:
        return self.source_frame_width / self.source_frame_height

    def compatibility_warning(self, width: int, height: int) -> str | None:
        if width <= 0 or height <= 0:
            return "Nieprawidłowy rozmiar klatki."
        aspect = width / height
        if abs(aspect / self.source_aspect_ratio - 1.0) > ASPECT_TOLERANCE:
            return (
                "Proporcje obrazu różnią się od kalibracji 3D "
                f"({self.source_frame_width}x{self.source_frame_height} vs {width}x{height})."
            )
        return None

    def camera_matrix_for(self, width: int, height: int) -> np.ndarray:
        warning = self.compatibility_warning(width, height)
        if warning:
            raise IntrinsicCalibrationError(warning)
        matrix = np.asarray(self.camera_matrix, dtype=np.float64).copy()
        sx = width / self.source_frame_width
        sy = height / self.source_frame_height
        matrix[0, 0] *= sx
        matrix[0, 2] *= sx
        matrix[1, 1] *= sy
        matrix[1, 2] *= sy
        return matrix

    def dist_array(self) -> np.ndarray:
        return np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1, 1)


class Depth3DCalibrationStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._items: dict[str, Depth3DCalibration] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            records = raw if isinstance(raw, list) else raw.get("calibrations", [])
            for record in records:
                calibration = Depth3DCalibration(**record)
                self._items[calibration.camera_id] = calibration
        except Exception as exc:
            print(f"[depth3d] failed to load calibration store: {exc}")

    def _save_locked(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = f"{self.path}.tmp"
        payload = {"calibrations": [asdict(item) for item in self._items.values()]}
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    def get(self, camera_id: str) -> Depth3DCalibration | None:
        with self._lock:
            item = self._items.get(camera_id)
            return Depth3DCalibration(**asdict(item)) if item else None

    def put(self, calibration: Depth3DCalibration) -> None:
        calibration.updated_at = time.time()
        with self._lock:
            self._items[calibration.camera_id] = calibration
            self._save_locked()

    def all(self) -> list[Depth3DCalibration]:
        with self._lock:
            return [Depth3DCalibration(**asdict(item)) for item in self._items.values()]


def make_observation(
    camera_id: str,
    timestamp: float,
    frame_bgr: np.ndarray,
    spec: CharucoBoardSpec,
) -> CharucoObservation:
    corners, ids, marker_corners, marker_ids = detect_charuco(frame_bgr, spec)
    corner_count = 0 if ids is None else int(len(ids))
    if corners is None or ids is None or corner_count < MIN_CAPTURE_CORNERS:
        raise IntrinsicCalibrationError(
            f"Wykryto {corner_count} narożników ChArUco; wymagane minimum to {MIN_CAPTURE_CORNERS}."
        )
    height, width = frame_bgr.shape[:2]
    return CharucoObservation(
        camera_id=camera_id,
        timestamp=float(timestamp),
        frame_width=int(width),
        frame_height=int(height),
        charuco_corners=np.asarray(corners, dtype=np.float32).copy(),
        charuco_ids=np.asarray(ids, dtype=np.int32).copy(),
        marker_count=0 if marker_ids is None else int(len(marker_ids)),
    )


def calibrate_intrinsics(
    camera_id: str,
    observations: list[CharucoObservation],
    spec: CharucoBoardSpec,
) -> Depth3DCalibration:
    if len(observations) < MIN_CALIBRATION_VIEWS:
        raise IntrinsicCalibrationError(
            f"Zapisz co najmniej {MIN_CALIBRATION_VIEWS} różnych widoków planszy."
        )
    width = observations[0].frame_width
    height = observations[0].frame_height
    for observation in observations:
        if (observation.frame_width, observation.frame_height) != (width, height):
            raise IntrinsicCalibrationError(
                "Wszystkie klatki kalibracyjne muszą mieć tę samą rozdzielczość."
            )

    board = create_charuco_board(spec)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    for observation in observations:
        obj, img = board.matchImagePoints(
            observation.charuco_corners,
            observation.charuco_ids,
        )
        if obj is None or img is None or len(obj) < MIN_CAPTURE_CORNERS:
            continue
        object_points.append(np.asarray(obj, dtype=np.float32).reshape(-1, 3))
        image_points.append(np.asarray(img, dtype=np.float32).reshape(-1, 2))

    if len(object_points) < MIN_CALIBRATION_VIEWS:
        raise IntrinsicCalibrationError("Za mało poprawnych widoków do kalibracji.")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        (width, height),
        None,
        None,
    )
    if not math.isfinite(float(rms)) or not np.all(np.isfinite(camera_matrix)):
        raise IntrinsicCalibrationError("Kalibracja zwróciła nieprawidłowe parametry.")

    errors: list[float] = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        errors.append(float(np.mean(np.linalg.norm(projected - img, axis=1))))
    mean_error = float(np.mean(errors)) if errors else float(rms)
    if mean_error > 2.5:
        raise IntrinsicCalibrationError(
            f"Błąd reprojekcji {mean_error:.2f}px jest zbyt duży. Zbierz ostrzejsze i bardziej zróżnicowane widoki."
        )

    return Depth3DCalibration(
        camera_id=camera_id,
        source_frame_width=width,
        source_frame_height=height,
        camera_matrix=camera_matrix.tolist(),
        dist_coeffs=np.asarray(dist_coeffs).reshape(-1).tolist(),
        rms_error_px=float(rms),
        mean_reprojection_error_px=mean_error,
        views_used=len(object_points),
        board_spec=asdict(spec),
    )


class MonocularDepthEstimator:
    """Lazy Hugging Face Depth Anything V2 metric-depth wrapper."""

    def __init__(
        self,
        *,
        enabled: bool,
        model_id: str,
        device: str,
        local_files_only: bool = False,
    ):
        self.enabled = bool(enabled)
        self.model_id = model_id
        self.device_spec = device
        self.local_files_only = bool(local_files_only)
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None
        self._load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def status(self) -> dict[str, Any]:
        device = str(self._device or self.device_spec)
        return {
            "enabled": self.enabled,
            "loaded": self.loaded,
            "model_id": self.model_id,
            "device": device,
            "precision": "fp16" if device.startswith("cuda") else "fp32",
            "available": self.enabled and self._load_error is None,
            "error": self._load_error,
        }

    def _resolve_device(self, torch):
        value = str(self.device_spec or "cpu").strip().lower()
        if value in {"0", "gpu", "cuda", "cuda:0"}:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if value.startswith("cuda"):
            return torch.device(value if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    def _ensure_loaded(self) -> None:
        if not self.enabled:
            raise Depth3DUnavailableError("Moduł 3D jest wyłączony w konfiguracji.")
        if self._model is not None:
            return
        if self._load_error:
            raise Depth3DUnavailableError(self._load_error)
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception as exc:
            self._load_error = (
                "Brak opcjonalnej biblioteki transformers. Uruchom: "
                "python -m pip install -r requirements-depth3d.txt"
            )
            raise Depth3DUnavailableError(self._load_error) from exc

        try:
            device = self._resolve_device(torch)
            processor = AutoImageProcessor.from_pretrained(
                self.model_id,
                local_files_only=self.local_files_only,
            )
            model = AutoModelForDepthEstimation.from_pretrained(
                self.model_id,
                local_files_only=self.local_files_only,
            )
            model.eval().to(device)
            if device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                model.half()
            self._torch = torch
            self._device = device
            self._processor = processor
            self._model = model
        except Exception as exc:
            self._load_error = f"Nie udało się wczytać modelu {self.model_id}: {exc}"
            raise Depth3DUnavailableError(self._load_error) from exc

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        with self._lock:
            self._ensure_loaded()
            assert self._torch is not None
            assert self._model is not None
            assert self._processor is not None
            from PIL import Image

            torch = self._torch
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {
                key: value.to(self._device)
                for key, value in inputs.items()
            }
            if self._device.type == "cuda":
                inputs = {
                    key: value.half() if value.is_floating_point() else value
                    for key, value in inputs.items()
                }
            with torch.inference_mode():
                if self._device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = self._model(**inputs)
                else:
                    output = self._model(**inputs)
                depth = output.predicted_depth.unsqueeze(1)
                depth = torch.nn.functional.interpolate(
                    depth,
                    size=frame_bgr.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            result = depth.float().cpu().numpy().astype(np.float32)
            result[~np.isfinite(result)] = np.nan
            return result


@dataclass(frozen=True)
class PersonDepthEstimate:
    box: tuple[int, int, int, int]
    confidence: float
    depth_z_m: float
    ray_distance_m: float
    valid_ratio: float
    sample_count: int
    anchor_pixel: tuple[float, float]


@dataclass
class DepthCorrectionModel:
    method: str
    scale: float
    offset: float
    point_count: int
    rmse_m: float
    median_abs_error_m: float


@dataclass
class Depth3DResult:
    camera_id: str
    timestamp: float
    frame_width: int
    frame_height: int
    depth_m: np.ndarray
    frame_bgr: np.ndarray
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    processing_ms: float
    model_id: str
    depth_scale: float
    raw_depth_m: np.ndarray | None = None
    world_from_camera: np.ndarray | None = None
    person_depths: list[PersonDepthEstimate] = field(default_factory=list)
    person_detection_ms: float = 0.0
    pipeline_ms: float = 0.0
    person_detection_warning: str | None = None
    correction_summary: dict[str, Any] | None = None


class Depth3DResultStore:
    def __init__(self, max_cameras: int = 8):
        self.max_cameras = max_cameras
        self._lock = threading.Lock()
        self._items: OrderedDict[str, Depth3DResult] = OrderedDict()

    def put(self, result: Depth3DResult) -> None:
        with self._lock:
            self._items.pop(result.camera_id, None)
            self._items[result.camera_id] = result
            while len(self._items) > self.max_cameras:
                self._items.popitem(last=False)

    def get(self, camera_id: str) -> Depth3DResult | None:
        with self._lock:
            item = self._items.get(camera_id)
            if item is None:
                return None
            return Depth3DResult(
                camera_id=item.camera_id,
                timestamp=item.timestamp,
                frame_width=item.frame_width,
                frame_height=item.frame_height,
                depth_m=item.depth_m.copy(),
                frame_bgr=item.frame_bgr.copy(),
                camera_matrix=item.camera_matrix.copy(),
                dist_coeffs=item.dist_coeffs.copy(),
                processing_ms=item.processing_ms,
                model_id=item.model_id,
                depth_scale=item.depth_scale,
                raw_depth_m=(None if item.raw_depth_m is None else item.raw_depth_m.copy()),
                world_from_camera=(
                    None if item.world_from_camera is None else item.world_from_camera.copy()
                ),
                person_depths=list(item.person_depths),
                person_detection_ms=item.person_detection_ms,
                pipeline_ms=item.pipeline_ms,
                person_detection_warning=item.person_detection_warning,
                correction_summary=(None if item.correction_summary is None else dict(item.correction_summary)),
            )


def depth_to_xyz(
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    stride: int = 1,
    min_depth_m: float = 0.1,
    max_depth_m: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    if stride < 1:
        raise ValueError("stride must be at least 1")
    height, width = depth_m.shape[:2]
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    z = depth_m[0:height:stride, 0:width:stride]
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)
    return points[valid].astype(np.float32), valid


def pixel_to_point(
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    point: tuple[float, float],
    *,
    radius: int = 3,
) -> np.ndarray:
    height, width = depth_m.shape[:2]
    u = int(round(point[0]))
    v = int(round(point[1]))
    if u < 0 or v < 0 or u >= width or v >= height:
        raise Depth3DError("Punkt znajduje się poza obrazem.")
    x1, x2 = max(0, u - radius), min(width, u + radius + 1)
    y1, y2 = max(0, v - radius), min(height, v + radius + 1)
    values = depth_m[y1:y2, x1:x2]
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        raise Depth3DError("Brak poprawnej głębi w wybranym punkcie.")
    z = float(np.median(values))
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    return np.asarray([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)


def estimate_person_depth(
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    confidence: float = 0.0,
    min_depth_m: float = 0.2,
    max_depth_m: float = 80.0,
) -> PersonDepthEstimate | None:
    """Estimate a person's distance from a robust torso-region depth sample.

    The central torso region avoids most bbox background, the floor and the gap
    between the legs. Median/MAD filtering makes the value much less sensitive
    to a single bad depth pixel than reading the bbox centre directly.
    """
    height, width = depth_m.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in box)
    x1, x2 = max(0, min(width - 1, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height - 1, y1)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    box_width = x2 - x1
    box_height = y2 - y1
    roi_x1 = max(x1, int(round(x1 + 0.25 * box_width)))
    roi_x2 = min(x2, int(round(x1 + 0.75 * box_width)))
    roi_y1 = max(y1, int(round(y1 + 0.15 * box_height)))
    roi_y2 = min(y2, int(round(y1 + 0.70 * box_height)))
    roi = depth_m[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return None

    valid_mask = (
        np.isfinite(roi)
        & (roi >= float(min_depth_m))
        & (roi <= float(max_depth_m))
    )
    values = roi[valid_mask].astype(np.float64, copy=False)
    valid_ratio = float(values.size / roi.size)
    if values.size < 20 or valid_ratio < 0.20:
        return None

    median = float(np.median(values))
    deviations = np.abs(values - median)
    mad = float(np.median(deviations))
    if mad > 1e-6:
        filtered = values[deviations <= 2.5 * mad]
        if filtered.size >= 20:
            values = filtered
            median = float(np.median(values))

    anchor_u = float((roi_x1 + roi_x2 - 1) / 2.0)
    anchor_v = float((roi_y1 + roi_y2 - 1) / 2.0)
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    x_m = (anchor_u - cx) * median / fx
    y_m = (anchor_v - cy) * median / fy
    ray_distance = float(math.sqrt(x_m * x_m + y_m * y_m + median * median))
    return PersonDepthEstimate(
        box=(x1, y1, x2, y2),
        confidence=float(confidence),
        depth_z_m=median,
        ray_distance_m=ray_distance,
        valid_ratio=valid_ratio,
        sample_count=int(values.size),
        anchor_pixel=(anchor_u, anchor_v),
    )


def transform_points(points: np.ndarray, transform: np.ndarray | None) -> np.ndarray:
    if transform is None or not len(points):
        return points
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)],
        axis=1,
    )
    return (homogeneous @ transform.T)[:, :3].astype(np.float32)


def depth_visualization_png(
    depth_m: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> bytes:
    clipped = np.clip(depth_m, min_depth_m, max_depth_m)
    normalized = (clipped - min_depth_m) / max(1e-6, max_depth_m - min_depth_m)
    normalized[~np.isfinite(normalized)] = 1.0
    image = np.uint8(np.clip((1.0 - normalized) * 255.0, 0, 255))
    colored = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    ok, encoded = cv2.imencode(".png", colored)
    if not ok:
        raise Depth3DError("Nie udało się zakodować mapy głębi.")
    return encoded.tobytes()


def depth_visualization_jpeg(
    depth_m: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
    quality: int = 84,
) -> bytes:
    clipped = np.clip(depth_m, min_depth_m, max_depth_m)
    normalized = (clipped - min_depth_m) / max(1e-6, max_depth_m - min_depth_m)
    normalized[~np.isfinite(normalized)] = 1.0
    image = np.uint8(np.clip((1.0 - normalized) * 255.0, 0, 255))
    colored = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    ok, encoded = cv2.imencode(
        ".jpg",
        colored,
        [cv2.IMWRITE_JPEG_QUALITY, int(max(40, min(95, quality)))],
    )
    if not ok:
        raise Depth3DError("Nie udało się zakodować mapy głębi JPEG.")
    return encoded.tobytes()


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 82) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, int(max(30, min(95, quality)))],
    )
    if not ok:
        raise Depth3DError("Nie udało się zakodować klatki RGB.")
    return encoded.tobytes()


def encode_sparse_cloud(
    points: np.ndarray,
    colors_bgr: np.ndarray,
    *,
    max_points: int = 4500,
) -> dict[str, Any]:
    if len(points) > max_points:
        indexes = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indexes]
        colors_bgr = colors_bgr[indexes]
    colors_rgb = colors_bgr[:, ::-1].astype(np.uint8, copy=False)
    return {
        "count": int(len(points)),
        "points_f32_b64": base64.b64encode(points.astype("<f4").tobytes()).decode("ascii"),
        "colors_u8_b64": base64.b64encode(colors_rgb.tobytes()).decode("ascii"),
    }


def pointcloud_ply(points: np.ndarray, colors_bgr: np.ndarray) -> bytes:
    colors_rgb = colors_bgr[:, ::-1].astype(np.uint8, copy=False)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    data = np.empty(len(points), dtype=dtype)
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["r"], data["g"], data["b"] = colors_rgb[:, 0], colors_rgb[:, 1], colors_rgb[:, 2]
    return header + data.tobytes()


def estimate_board_pose(
    frame_bgr: np.ndarray,
    calibration: Depth3DCalibration,
    spec: CharucoBoardSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = frame_bgr.shape[:2]
    camera_matrix = calibration.camera_matrix_for(width, height)
    dist_coeffs = calibration.dist_array()
    corners, ids, _, _ = detect_charuco(
        frame_bgr,
        spec,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    if corners is None or ids is None or len(ids) < MIN_CAPTURE_CORNERS:
        raise IntrinsicCalibrationError(
            f"Do ustawienia układu świata wymagane jest minimum {MIN_CAPTURE_CORNERS} narożników ChArUco."
        )
    board = create_charuco_board(spec)
    obj, img = board.matchImagePoints(corners, ids)
    success, rvec, tvec = cv2.solvePnP(
        np.asarray(obj, dtype=np.float32),
        np.asarray(img, dtype=np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise IntrinsicCalibrationError("Nie udało się wyznaczyć pozy planszy.")
    return rvec, tvec, np.asarray(obj, dtype=np.float64).reshape(-1, 3), np.asarray(img, dtype=np.float64).reshape(-1, 2)


def world_from_board_pose(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(rvec)
    camera_from_world = np.eye(4, dtype=np.float64)
    camera_from_world[:3, :3] = rotation
    camera_from_world[:3, 3] = np.asarray(tvec).reshape(3)
    return np.linalg.inv(camera_from_world)


def fit_depth_scale(
    depth_m: np.ndarray,
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[float, float, int]:
    rotation, _ = cv2.Rodrigues(rvec)
    camera_points = (rotation @ object_points.T).T + np.asarray(tvec).reshape(1, 3)
    ratios: list[float] = []
    errors: list[float] = []
    pairs: list[tuple[float, float]] = []
    height, width = depth_m.shape[:2]
    for true_point, image_point in zip(camera_points, image_points):
        u, v = int(round(image_point[0])), int(round(image_point[1]))
        if not (0 <= u < width and 0 <= v < height):
            continue
        y1, y2 = max(0, v - 2), min(height, v + 3)
        x1, x2 = max(0, u - 2), min(width, u + 3)
        values = depth_m[y1:y2, x1:x2]
        values = values[np.isfinite(values) & (values > 0.05)]
        if not values.size or true_point[2] <= 0:
            continue
        predicted = float(np.median(values))
        true_z = float(true_point[2])
        ratios.append(true_z / predicted)
        pairs.append((predicted, true_z))
    if len(ratios) < 6:
        raise IntrinsicCalibrationError("Za mało poprawnych punktów do korekcji skali głębi.")
    scale = float(np.median(ratios))
    scale = max(0.25, min(4.0, scale))
    for predicted, true_z in pairs:
        errors.append(abs(predicted * scale - true_z))
    return scale, float(np.median(errors)), len(pairs)



def _depth_fit_metrics(predicted: np.ndarray, measured: np.ndarray, estimated: np.ndarray) -> tuple[float, float]:
    residual = estimated - measured
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    medae = float(np.median(np.abs(residual)))
    return rmse, medae


def fit_depth_correction_model(
    predicted_depths: list[float] | np.ndarray,
    measured_depths: list[float] | np.ndarray,
    *,
    preferred_method: str = "auto",
) -> DepthCorrectionModel:
    predicted = np.asarray(predicted_depths, dtype=np.float64).reshape(-1)
    measured = np.asarray(measured_depths, dtype=np.float64).reshape(-1)
    valid = (np.isfinite(predicted) & np.isfinite(measured) & (predicted > 0.05) & (measured > 0.05))
    predicted = predicted[valid]
    measured = measured[valid]
    if predicted.size < 3:
        raise IntrinsicCalibrationError("Do korekcji głębi potrzebne są co najmniej 3 poprawne punkty kontrolne.")

    candidates: list[DepthCorrectionModel] = []
    preferred_method = str(preferred_method or "auto").strip().lower()

    if preferred_method in {"auto", "affine"}:
        design = np.column_stack([predicted, np.ones_like(predicted)])
        scale, offset = np.linalg.lstsq(design, measured, rcond=None)[0]
        scale = float(np.clip(scale, 0.05, 20.0))
        offset = float(np.clip(offset, -50.0, 50.0))
        estimated = scale * predicted + offset
        estimated = np.maximum(estimated, 0.05)
        rmse, medae = _depth_fit_metrics(predicted, measured, estimated)
        candidates.append(DepthCorrectionModel(
            method="affine",
            scale=scale,
            offset=offset,
            point_count=int(predicted.size),
            rmse_m=rmse,
            median_abs_error_m=medae,
        ))

    if preferred_method in {"auto", "inverse_affine", "inverse"}:
        inv = 1.0 / np.maximum(predicted, 1e-4)
        design = np.column_stack([inv, np.ones_like(inv)])
        a, b = np.linalg.lstsq(design, 1.0 / np.maximum(measured, 1e-4), rcond=None)[0]
        a = float(np.clip(a, 0.01, 100.0))
        b = float(np.clip(b, -100.0, 100.0))
        denom = a * inv + b
        denom = np.where(denom > 1e-6, denom, np.nan)
        estimated = 1.0 / denom
        finite = np.isfinite(estimated) & (estimated > 0.05)
        if np.count_nonzero(finite) >= 3:
            rmse, medae = _depth_fit_metrics(predicted[finite], measured[finite], estimated[finite])
            candidates.append(DepthCorrectionModel(
                method="inverse_affine",
                scale=a,
                offset=b,
                point_count=int(np.count_nonzero(finite)),
                rmse_m=rmse,
                median_abs_error_m=medae,
            ))

    if not candidates:
        raise IntrinsicCalibrationError("Nie udało się dopasować korekcji głębi do podanych punktów.")
    candidates.sort(key=lambda item: (item.rmse_m, item.median_abs_error_m))
    if preferred_method == "affine":
        for item in candidates:
            if item.method == "affine":
                return item
    if preferred_method in {"inverse_affine", "inverse"}:
        for item in candidates:
            if item.method == "inverse_affine":
                return item
    return candidates[0]


def apply_depth_correction(depth_m: np.ndarray, correction: dict[str, Any] | DepthCorrectionModel | None) -> np.ndarray:
    if correction is None:
        return depth_m.copy()
    if isinstance(correction, DepthCorrectionModel):
        method = correction.method
        scale = float(correction.scale)
        offset = float(correction.offset)
    else:
        method = str(correction.get("method", "affine"))
        scale = float(correction.get("scale", 1.0))
        offset = float(correction.get("offset", 0.0))

    corrected = depth_m.astype(np.float32, copy=True)
    valid = np.isfinite(corrected) & (corrected > 0.0)
    if method == "inverse_affine":
        safe = np.maximum(corrected[valid].astype(np.float64), 1e-4)
        denom = scale * (1.0 / safe) + offset
        out = np.full_like(safe, np.nan, dtype=np.float64)
        good = denom > 1e-6
        out[good] = 1.0 / denom[good]
        corrected[valid] = out.astype(np.float32)
    else:
        corrected[valid] = (scale * corrected[valid].astype(np.float64) + offset).astype(np.float32)
    corrected[~np.isfinite(corrected)] = np.nan
    corrected[corrected <= 0] = np.nan
    return corrected
