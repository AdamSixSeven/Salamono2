"""API for persistent checkerboard profiles and real-time monocular depth."""
from __future__ import annotations

import asyncio
import base64
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field, field_validator

from backend.depth3d import (
    Depth3DError,
    Depth3DResult,
    Depth3DUnavailableError,
    IntrinsicCalibrationError,
    apply_depth_correction,
    depth_to_xyz,
    fit_depth_correction_model,
    depth_visualization_jpeg,
    encode_jpeg,
    estimate_person_depth,
    pixel_to_point,
    pointcloud_ply,
)
from backend.depth3d_checkerboard import (
    CheckerboardSpec,
    approximate_camera_matrix,
    calibrate_checkerboard_intrinsics,
    detect_checkerboard,
    make_checkerboard_observation,
    render_checkerboard_png,
    undistort_with_calibration,
)
from config import CONFIG

router = APIRouter(prefix="/depth3d", tags=["depth3d"])
LATEST_FRAME_MAX_AGE_SECONDS = 3.0
MIN_CALIBRATION_VIEWS = 8


class CheckerboardSpecIn(BaseModel):
    inner_corners_x: int = Field(default=9, ge=3, le=20)
    inner_corners_y: int = Field(default=6, ge=3, le=20)
    square_length_m: float = Field(default=0.03, gt=0, le=1)
    lens_model: str = Field(default="pinhole")

    @field_validator("lens_model")
    @classmethod
    def valid_lens_model(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"pinhole", "fisheye"}:
            raise ValueError("lens_model must be pinhole or fisheye")
        return value

    def to_spec(self) -> CheckerboardSpec:
        return CheckerboardSpec(**self.model_dump()).validate()


class ProfileCreateIn(BaseModel):
    name: str = Field(default="Nowy profil", min_length=1, max_length=120)
    checkerboard: CheckerboardSpecIn = Field(default_factory=CheckerboardSpecIn)


class ProfileRenameIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class DepthPointIn(BaseModel):
    point: list[float] = Field(min_length=2, max_length=2)
    radius: int = Field(default=5, ge=1, le=25)

    @field_validator("point")
    @classmethod
    def finite_point(cls, value: list[float]) -> list[float]:
        if not all(math.isfinite(float(item)) for item in value):
            raise ValueError("coordinates must be finite")
        return [float(item) for item in value]


class Measure3DIn(BaseModel):
    point_a: list[float] = Field(min_length=2, max_length=2)
    point_b: list[float] = Field(min_length=2, max_length=2)


class DepthMetricPointIn(BaseModel):
    point: list[float] = Field(min_length=2, max_length=2)
    measured_distance_m: float = Field(gt=0.05, le=200.0)
    label: str | None = Field(default=None, max_length=80)
    radius: int = Field(default=6, ge=1, le=25)

    @field_validator("point")
    @classmethod
    def finite_point(cls, value: list[float]) -> list[float]:
        if not all(math.isfinite(float(item)) for item in value):
            raise ValueError("coordinates must be finite")
        return [float(item) for item in value]


class DepthMetricFitIn(BaseModel):
    method: str = Field(default="auto")

    @field_validator("method")
    @classmethod
    def valid_method(cls, value: str) -> str:
        value = str(value or "auto").strip().lower()
        if value not in {"auto", "affine", "inverse_affine", "inverse"}:
            raise ValueError("method must be auto, affine or inverse_affine")
        return value


def _latest_frame(request: Request, camera_id: str):
    item = request.app.state.latest_frame_store.get(camera_id, max_age_seconds=LATEST_FRAME_MAX_AGE_SECONDS)
    if item is None:
        raise HTTPException(409, "Brak świeżej klatki tej kamery. Uruchom wysyłanie obrazu i spróbuj ponownie.")
    return item


def _profiles(request: Request):
    store = getattr(request.app.state, "depth3d_profile_store", None)
    if store is None:
        raise HTTPException(409, "Magazyn profili kalibracyjnych nie jest dostępny.")
    return store


def _calibration_out(calibration):
    if calibration is None:
        return None
    result = asdict(calibration)
    result["source_aspect_ratio"] = calibration.source_aspect_ratio
    result["pattern_type"] = calibration.board_spec.get("pattern_type", "legacy")
    result["lens_model"] = calibration.board_spec.get("lens_model", "pinhole")
    return result


def _depth_stats(depth: np.ndarray) -> tuple[float | None, float | None, float | None]:
    values = depth[np.isfinite(depth) & (depth > 0)]
    if not values.size:
        return None, None, None
    return round(float(np.percentile(values, 2)), 3), round(float(np.median(values)), 3), round(float(np.percentile(values, 98)), 3)


@router.get("/status")
async def status(request: Request, camera_id: str = "cam_default"):
    store = _profiles(request)
    active_id = store.ensure_active(camera_id)
    profile = store.get_profile(camera_id, active_id)
    calibration = store.get_calibration(camera_id, active_id)
    captures = store.get_observations(camera_id, active_id)
    latest = request.app.state.latest_frame_store.get(camera_id)
    result = request.app.state.depth3d_result_store.get(camera_id)
    warning = None
    if calibration is not None and latest is not None:
        warning = calibration.compatibility_warning(latest.width, latest.height)
    return {
        "camera_id": camera_id,
        "frame_available": latest is not None,
        "frame_age_sec": None if latest is None else round(time.time() - latest.received_at, 3),
        "frame_width": None if latest is None else latest.width,
        "frame_height": None if latest is None else latest.height,
        "captures": len(captures),
        "minimum_captures": MIN_CALIBRATION_VIEWS,
        "active_profile_id": active_id,
        "active_profile": profile,
        "profiles": store.list_profiles(camera_id),
        "calibration": _calibration_out(calibration),
        "calibration_compatible": calibration is not None and warning is None,
        "compatibility_warning": warning,
        "model": request.app.state.depth3d_estimator.status(),
        "target_fps": CONFIG.depth3d.target_fps,
        "latest_result": None if result is None else {
            "timestamp": result.timestamp,
            "age_sec": round(time.time() - result.timestamp, 3),
            "processing_ms": round(result.processing_ms, 2),
            "pipeline_ms": round(result.pipeline_ms, 2),
            "person_count": len(result.person_depths),
            "inference_fps": round(1000.0 / max(0.001, result.processing_ms), 2),
            "median_depth_m": _depth_stats(result.depth_m)[1],
        },
        "limitations": "Monokularna głębia jest estymacją sieci neuronowej, nie pomiarem stereo/ToF/LiDAR.",
    }


@router.get("/checkerboard.png")
async def checkerboard_png(
    inner_corners_x: int = Query(9, ge=3, le=20),
    inner_corners_y: int = Query(6, ge=3, le=20),
    square_length_m: float = Query(0.03, gt=0, le=1),
    dpi: int = Query(300, ge=72, le=1200),
):
    spec = CheckerboardSpec(inner_corners_x, inner_corners_y, square_length_m).validate()
    payload = render_checkerboard_png(spec, dpi=dpi)
    return Response(payload, media_type="image/png", headers={"Content-Disposition": 'attachment; filename="checkerboard-calibration.png"'})


@router.get("/{camera_id}/profiles")
async def list_profiles(camera_id: str, request: Request):
    store = _profiles(request)
    store.ensure_active(camera_id)
    return {"camera_id": camera_id, "active_profile_id": store.active_profile_id(camera_id), "profiles": store.list_profiles(camera_id)}


@router.post("/{camera_id}/profiles")
async def create_profile(camera_id: str, body: ProfileCreateIn, request: Request):
    return _profiles(request).create_profile(camera_id, body.name, body.checkerboard.to_spec(), activate=True)


@router.post("/{camera_id}/profiles/{profile_id}/activate")
async def activate_profile(camera_id: str, profile_id: str, request: Request):
    try:
        return _profiles(request).activate(camera_id, profile_id)
    except KeyError as exc:
        raise HTTPException(404, "Nie znaleziono profilu kalibracyjnego.") from exc


@router.patch("/{camera_id}/profiles/{profile_id}")
async def rename_profile(camera_id: str, profile_id: str, body: ProfileRenameIn, request: Request):
    try:
        return _profiles(request).rename(camera_id, profile_id, body.name)
    except KeyError as exc:
        raise HTTPException(404, "Nie znaleziono profilu kalibracyjnego.") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/{camera_id}/profiles/{profile_id}/export.zip")
async def export_profile(camera_id: str, profile_id: str, request: Request):
    try:
        payload = _profiles(request).export_profile_zip(camera_id, profile_id)
    except KeyError as exc:
        raise HTTPException(404, "Nie znaleziono profilu kalibracyjnego.") from exc
    filename = f"depth3d-{camera_id}-{profile_id}.zip".replace("/", "_").replace("\\", "_")
    return Response(payload, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.delete("/{camera_id}/profiles/{profile_id}")
async def delete_profile(camera_id: str, profile_id: str, request: Request):
    try:
        _profiles(request).delete_profile(camera_id, profile_id)
    except KeyError as exc:
        raise HTTPException(404, "Nie znaleziono profilu kalibracyjnego.") from exc
    store = _profiles(request)
    active = store.ensure_active(camera_id)
    return {"camera_id": camera_id, "deleted_profile_id": profile_id, "active_profile_id": active}


@router.get("/{camera_id}/profiles/{profile_id}/views")
async def list_profile_views(camera_id: str, profile_id: str, request: Request):
    return {"camera_id": camera_id, "profile_id": profile_id, "views": _profiles(request).list_views(camera_id, profile_id)}


@router.get("/{camera_id}/profiles/{profile_id}/views/{view_id}/image")
async def profile_view_image(camera_id: str, profile_id: str, view_id: str, request: Request):
    path = _profiles(request).image_path(camera_id, profile_id, view_id)
    if path is None:
        raise HTTPException(404, "Nie znaleziono obrazu widoku kalibracyjnego.")
    return Response(path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.delete("/{camera_id}/profiles/{profile_id}/views/{view_id}")
async def delete_profile_view(camera_id: str, profile_id: str, view_id: str, request: Request):
    try:
        _profiles(request).delete_view(camera_id, profile_id, view_id)
    except KeyError as exc:
        raise HTTPException(404, "Nie znaleziono widoku kalibracyjnego.") from exc
    return {"camera_id": camera_id, "profile_id": profile_id, "deleted_view_id": view_id}


@router.post("/{camera_id}/checkerboard/detect")
async def checkerboard_detect(camera_id: str, body: CheckerboardSpecIn, request: Request):
    item = _latest_frame(request, camera_id)
    corners, diagnostics = await asyncio.to_thread(detect_checkerboard, item.frame, body.to_spec())
    diagnostics.update({"camera_id": camera_id, "frame_width": item.width, "frame_height": item.height})
    return diagnostics


@router.post("/{camera_id}/checkerboard/capture")
async def checkerboard_capture(camera_id: str, body: CheckerboardSpecIn, request: Request):
    item = _latest_frame(request, camera_id)
    store = _profiles(request)
    profile_id = store.ensure_active(camera_id, body.to_spec())
    try:
        observation = await asyncio.to_thread(make_checkerboard_observation, camera_id, item.timestamp, item.frame, body.to_spec())
        added, warning, view = await asyncio.to_thread(store.add_view, camera_id, observation, item.frame, body.to_spec(), profile_id)
    except IntrinsicCalibrationError as exc:
        raise HTTPException(422, str(exc)) from exc
    captures = store.get_observations(camera_id, profile_id)
    return {
        "camera_id": camera_id,
        "profile_id": profile_id,
        "view": view or None,
        "accepted": added,
        "warning": warning,
        "capture_count": len(captures),
        "corner_count": observation.corner_count,
        "expected_corners": body.inner_corners_x * body.inner_corners_y,
        "coverage_ratio": round(observation.coverage_ratio, 4),
        "blur_score": round(observation.blur_score, 1),
        "frame_width": observation.frame_width,
        "frame_height": observation.frame_height,
    }


@router.delete("/{camera_id}/captures")
async def reset_captures(camera_id: str, request: Request):
    store = _profiles(request)
    profile_id = store.ensure_active(camera_id)
    store.clear_views(camera_id, profile_id)
    return {"camera_id": camera_id, "profile_id": profile_id, "capture_count": 0}


@router.post("/{camera_id}/checkerboard/calibrate")
async def checkerboard_calibrate(camera_id: str, body: CheckerboardSpecIn, request: Request):
    store = _profiles(request)
    profile_id = store.ensure_active(camera_id, body.to_spec())
    captures = store.get_observations(camera_id, profile_id)
    try:
        calibration = await asyncio.to_thread(calibrate_checkerboard_intrinsics, camera_id, captures, body.to_spec())
    except IntrinsicCalibrationError as exc:
        raise HTTPException(422, str(exc)) from exc
    store.save_calibration(camera_id, profile_id, calibration)
    result = _calibration_out(calibration)
    result["profile_id"] = profile_id
    return result


@router.post("/{camera_id}/profiles/{profile_id}/calculate")
async def calculate_profile(camera_id: str, profile_id: str, request: Request):
    store = _profiles(request)
    profile = store.get_profile(camera_id, profile_id)
    if not profile:
        raise HTTPException(404, "Nie znaleziono profilu kalibracyjnego.")
    try:
        raw_spec = profile["checkerboard_spec"]
        spec = CheckerboardSpec(**{
            key: raw_spec[key]
            for key in ("inner_corners_x", "inner_corners_y", "square_length_m", "lens_model")
            if key in raw_spec
        }).validate()
        calibration = await asyncio.to_thread(calibrate_checkerboard_intrinsics, camera_id, store.get_observations(camera_id, profile_id), spec)
        store.save_calibration(camera_id, profile_id, calibration)
    except IntrinsicCalibrationError as exc:
        raise HTTPException(422, str(exc)) from exc
    result = _calibration_out(calibration)
    result["profile_id"] = profile_id
    return result


@router.get("/{camera_id}/profiles/{profile_id}/depth-points")
async def list_depth_metric_points(camera_id: str, profile_id: str, request: Request):
    store = _profiles(request)
    profile = store.get_profile(camera_id, profile_id)
    if not profile:
        raise HTTPException(404, "Nie znaleziono profilu kalibracyjnego.")
    return {
        "camera_id": camera_id,
        "profile_id": profile_id,
        "points": store.list_depth_metric_points(camera_id, profile_id),
        "correction": store.get_depth_metric_correction(camera_id, profile_id),
    }


@router.post("/{camera_id}/profiles/{profile_id}/depth-points")
async def create_depth_metric_point(camera_id: str, profile_id: str, body: DepthMetricPointIn, request: Request):
    store = _profiles(request)
    profile = store.get_profile(camera_id, profile_id)
    if not profile:
        raise HTTPException(404, "Nie znaleziono profilu kalibracyjnego.")
    result = request.app.state.depth3d_result_store.get(camera_id)
    if result is None:
        raise HTTPException(409, "Najpierw uruchom inferencję głębi dla tej kamery.")
    raw_depth = result.raw_depth_m if result.raw_depth_m is not None else result.depth_m
    try:
        point_camera = pixel_to_point(raw_depth, result.camera_matrix, tuple(body.point), radius=body.radius)
    except Depth3DError as exc:
        raise HTTPException(422, str(exc)) from exc
    point = store.add_depth_metric_point(
        camera_id,
        profile_id,
        x=body.point[0],
        y=body.point[1],
        measured_distance_m=body.measured_distance_m,
        predicted_depth_m=float(point_camera[2]),
        label=body.label,
        frame_width=result.frame_width,
        frame_height=result.frame_height,
    )
    return {
        "camera_id": camera_id,
        "profile_id": profile_id,
        "point": point,
        "predicted_depth_m": round(float(point_camera[2]), 4),
        "predicted_ray_distance_m": round(float(np.linalg.norm(point_camera)), 4),
        "message": "Punkt kontrolny zapisany. Po dodaniu kilku punktów uruchom dopasowanie korekcji.",
    }


@router.delete("/{camera_id}/profiles/{profile_id}/depth-points/{point_id}")
async def delete_depth_metric_point(camera_id: str, profile_id: str, point_id: str, request: Request):
    try:
        _profiles(request).delete_depth_metric_point(camera_id, profile_id, point_id)
    except KeyError as exc:
        raise HTTPException(404, "Nie znaleziono punktu kontrolnego.") from exc
    return {"camera_id": camera_id, "profile_id": profile_id, "deleted_point_id": point_id}


@router.delete("/{camera_id}/profiles/{profile_id}/depth-points")
async def clear_depth_metric_points(camera_id: str, profile_id: str, request: Request):
    try:
        _profiles(request).clear_depth_metric_points(camera_id, profile_id)
    except KeyError as exc:
        raise HTTPException(404, "Nie znaleziono profilu kalibracyjnego.") from exc
    return {"camera_id": camera_id, "profile_id": profile_id, "points": []}


@router.post("/{camera_id}/profiles/{profile_id}/depth-correction/fit")
async def fit_depth_metric_correction(camera_id: str, profile_id: str, body: DepthMetricFitIn, request: Request):
    store = _profiles(request)
    profile = store.get_profile(camera_id, profile_id)
    if not profile:
        raise HTTPException(404, "Nie znaleziono profilu kalibracyjnego.")
    points = store.list_depth_metric_points(camera_id, profile_id)
    if len(points) < 3:
        raise HTTPException(422, "Dodaj co najmniej 3 punkty kontrolne z rzeczywistą odległością.")
    try:
        model = fit_depth_correction_model(
            [float(item.get("predicted_depth_m", 0.0)) for item in points],
            [float(item.get("measured_distance_m", 0.0)) for item in points],
            preferred_method=body.method,
        )
    except IntrinsicCalibrationError as exc:
        raise HTTPException(422, str(exc)) from exc
    correction = {
        "method": model.method,
        "scale": round(float(model.scale), 8),
        "offset": round(float(model.offset), 8),
        "point_count": int(model.point_count),
        "rmse_m": round(float(model.rmse_m), 6),
        "median_abs_error_m": round(float(model.median_abs_error_m), 6),
        "created_at": time.time(),
    }
    store.save_depth_metric_correction(camera_id, profile_id, correction)
    return {
        "camera_id": camera_id,
        "profile_id": profile_id,
        "correction": correction,
        "points": points,
    }


# Compatibility aliases used by early builds of the 3D page.
@router.post("/{camera_id}/capture", deprecated=True)
async def capture_alias(camera_id: str, body: CheckerboardSpecIn, request: Request):
    return await checkerboard_capture(camera_id, body, request)


@router.post("/{camera_id}/calibrate", deprecated=True)
async def calibrate_alias(camera_id: str, body: CheckerboardSpecIn, request: Request):
    return await checkerboard_calibrate(camera_id, body, request)


def _run_inference(request: Request, camera_id: str):
    pipeline_started = time.perf_counter()
    item = _latest_frame(request, camera_id)
    calibration = request.app.state.depth3d_calibration_store.get(camera_id)
    warning = None
    calibrated = False
    frame = item.frame
    if calibration is not None:
        warning = calibration.compatibility_warning(item.width, item.height)
        if warning is None:
            frame, camera_matrix = undistort_with_calibration(item.frame, calibration)
            calibrated = True
        else:
            camera_matrix = approximate_camera_matrix(item.width, item.height)
    else:
        camera_matrix = approximate_camera_matrix(item.width, item.height)
        warning = "Brak kalibracji intrinsics — mapa głębi działa na surowym obrazie."

    depth_started = time.perf_counter()
    # Depth Anything and live YOLO share the same CUDA device on the pilot
    # machine. Serializing the two large models prevents VRAM spikes and
    # unpredictable implicit stream synchronization. The endpoint already runs
    # in ``asyncio.to_thread``, so waiting here never blocks the event loop.
    analysis_lock = getattr(request.app.state, "analysis_lock", None)
    lock_context = analysis_lock if analysis_lock is not None else nullcontext()
    with lock_context:
        raw_depth = request.app.state.depth3d_estimator.infer(frame)
    processing_ms = (time.perf_counter() - depth_started) * 1000.0
    raw_depth[(raw_depth < CONFIG.depth3d.min_depth_m) | (raw_depth > CONFIG.depth3d.max_depth_m)] = np.nan

    profile_store = getattr(request.app.state, "depth3d_profile_store", None)
    active_profile_id = profile_store.active_profile_id(camera_id) if profile_store is not None else None
    correction = profile_store.get_depth_metric_correction(camera_id, active_profile_id) if profile_store is not None and active_profile_id else None
    depth = apply_depth_correction(raw_depth, correction) if correction else raw_depth.copy()
    depth[(depth < CONFIG.depth3d.min_depth_m) | (depth > CONFIG.depth3d.max_depth_m)] = np.nan

    person_depths = []
    persons = []
    person_detection_ms = 0.0
    person_detection_warning = None
    if CONFIG.depth3d.person_distance_enabled:
        detector = getattr(request.app.state, "detector", None)
        if detector is None:
            person_detection_warning = "Detektor YOLO nie jest dostępny."
        else:
            try:
                detection_started = time.perf_counter()
                detection_store = getattr(
                    request.app.state,
                    "latest_detection_store",
                    None,
                )
                cached = (
                    detection_store.get_exact(
                        camera_id,
                        item.timestamp,
                        item.width,
                        item.height,
                    )
                    if detection_store is not None else None
                )
                if cached is not None:
                    detections = list(cached.detections)
                else:
                    analysis_lock = getattr(request.app.state, "analysis_lock", None)
                    lock_context = analysis_lock if analysis_lock is not None else nullcontext()
                    with lock_context:
                        detections = detector.detect(frame)
                person_detection_ms = (time.perf_counter() - detection_started) * 1000.0
                persons = [detection for detection in detections if detection.category == "person"]
                persons.sort(key=lambda detection: (detection.box[0], detection.box[1]))
                for person in persons:
                    estimate = estimate_person_depth(
                        depth,
                        camera_matrix,
                        tuple(person.box),
                        confidence=person.confidence,
                        min_depth_m=CONFIG.depth3d.min_depth_m,
                        max_depth_m=CONFIG.depth3d.max_depth_m,
                    )
                    if estimate is not None:
                        person_depths.append(estimate)
            except Exception as exc:
                person_detection_warning = (
                    f"Pomiar osób niedostępny: {type(exc).__name__}: {exc}"
                )

    performance_profiler = getattr(
        request.app.state,
        "performance_profiler",
        None,
    )
    if performance_profiler is not None:
        performance_profiler.observe(
            "depth_ms",
            processing_ms,
            has_person=(
                bool(persons)
                if CONFIG.depth3d.person_distance_enabled else None
            ),
            source="depth3d",
        )

    result = Depth3DResult(
        camera_id=camera_id,
        timestamp=item.timestamp,
        frame_width=item.width,
        frame_height=item.height,
        depth_m=depth,
        frame_bgr=frame,
        camera_matrix=camera_matrix,
        dist_coeffs=np.zeros((5, 1), dtype=np.float64),
        processing_ms=processing_ms,
        model_id=request.app.state.depth3d_estimator.model_id,
        depth_scale=1.0,
        raw_depth_m=raw_depth,
        world_from_camera=None,
        person_depths=person_depths,
        correction_summary=(None if not correction else dict(correction)),
        person_detection_ms=person_detection_ms,
        pipeline_ms=(time.perf_counter() - pipeline_started) * 1000.0,
        person_detection_warning=person_detection_warning,
    )
    request.app.state.depth3d_result_store.put(result)
    return result, calibrated, warning


@router.post("/{camera_id}/infer")
async def infer(camera_id: str, request: Request):
    try:
        result, calibrated, warning = await asyncio.to_thread(_run_inference, request, camera_id)
    except (Depth3DUnavailableError, IntrinsicCalibrationError) as exc:
        raise HTTPException(409, str(exc)) from exc
    except Depth3DError as exc:
        raise HTTPException(422, str(exc)) from exc

    min_depth, median_depth, max_depth = _depth_stats(result.depth_m)
    encoding_started = time.perf_counter()
    rgb_bytes, depth_bytes = await asyncio.gather(
        asyncio.to_thread(encode_jpeg, result.frame_bgr, CONFIG.depth3d.jpeg_quality),
        asyncio.to_thread(
            depth_visualization_jpeg,
            result.depth_m,
            min_depth_m=CONFIG.depth3d.min_depth_m,
            max_depth_m=CONFIG.depth3d.visualization_max_depth_m,
            quality=CONFIG.depth3d.jpeg_quality,
        ),
    )
    encoding_ms = (time.perf_counter() - encoding_started) * 1000.0
    backend_total_ms = result.pipeline_ms + encoding_ms
    person_distances = [
        {
            "person_index": index,
            "box": list(person.box),
            "confidence": round(person.confidence, 4),
            "depth_z_m": round(person.depth_z_m, 4),
            "ray_distance_m": round(person.ray_distance_m, 4),
            "valid_ratio": round(person.valid_ratio, 4),
            "sample_count": person.sample_count,
            "anchor_pixel": [round(value, 2) for value in person.anchor_pixel],
        }
        for index, person in enumerate(result.person_depths, start=1)
    ]
    return {
        "camera_id": camera_id,
        "timestamp": result.timestamp,
        "frame_width": result.frame_width,
        "frame_height": result.frame_height,
        "processing_ms": round(result.processing_ms, 2),
        "inference_fps": round(1000.0 / max(0.001, result.processing_ms), 2),
        "person_detection_ms": round(result.person_detection_ms, 2),
        "pipeline_ms": round(result.pipeline_ms, 2),
        "encoding_ms": round(encoding_ms, 2),
        "backend_total_ms": round(backend_total_ms, 2),
        "person_detection_warning": result.person_detection_warning,
        "person_distances": person_distances,
        "model_id": result.model_id,
        "device": request.app.state.depth3d_estimator.status().get("device"),
        "correction_applied": result.correction_summary is not None,
        "correction_summary": result.correction_summary,
        "calibrated": calibrated,
        "calibration_warning": warning,
        "min_depth_m": min_depth,
        "median_depth_m": median_depth,
        "max_depth_m": max_depth,
        "rgb_jpeg_b64": base64.b64encode(rgb_bytes).decode("ascii"),
        "depth_jpeg_b64": base64.b64encode(depth_bytes).decode("ascii"),
        "depth_image_mime": "image/jpeg",
    }


@router.post("/{camera_id}/distance")
async def distance_from_camera(camera_id: str, body: DepthPointIn, request: Request):
    result = request.app.state.depth3d_result_store.get(camera_id)
    if result is None:
        raise HTTPException(409, "Najpierw uruchom inferencję głębi dla tej kamery.")
    try:
        point = pixel_to_point(result.depth_m, result.camera_matrix, tuple(body.point), radius=body.radius)
    except Depth3DError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "camera_id": camera_id,
        "pixel": [round(float(body.point[0]), 2), round(float(body.point[1]), 2)],
        "depth_z_m": round(float(point[2]), 4),
        "ray_distance_m": round(float(np.linalg.norm(point)), 4),
        "point_camera_m": [round(float(value), 4) for value in point],
        "result_age_sec": round(time.time() - result.timestamp, 3),
    }


@router.post("/{camera_id}/measure")
async def measure(camera_id: str, body: Measure3DIn, request: Request):
    result = request.app.state.depth3d_result_store.get(camera_id)
    if result is None:
        raise HTTPException(409, "Najpierw uruchom inferencję głębi dla tej kamery.")
    point_a = pixel_to_point(result.depth_m, result.camera_matrix, tuple(body.point_a))
    point_b = pixel_to_point(result.depth_m, result.camera_matrix, tuple(body.point_b))
    return {
        "camera_id": camera_id,
        "point_a_m": [round(float(v), 4) for v in point_a],
        "point_b_m": [round(float(v), 4) for v in point_b],
        "distance_m": round(float(np.linalg.norm(point_b - point_a)), 4),
    }


@router.get("/{camera_id}/pointcloud.ply")
async def download_pointcloud(camera_id: str, request: Request):
    result = request.app.state.depth3d_result_store.get(camera_id)
    if result is None:
        raise HTTPException(409, "Najpierw uruchom inferencję głębi dla tej kamery.")
    points, valid = depth_to_xyz(result.depth_m, result.camera_matrix, stride=CONFIG.depth3d.export_point_stride, min_depth_m=CONFIG.depth3d.min_depth_m, max_depth_m=CONFIG.depth3d.max_depth_m)
    colors = result.frame_bgr[0:result.frame_height:CONFIG.depth3d.export_point_stride, 0:result.frame_width:CONFIG.depth3d.export_point_stride][valid]
    return Response(pointcloud_ply(points, colors), media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="{camera_id}-pointcloud.ply"'})
