"""Calibration API for uploaded images and live camera streams."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)

from backend.calibration import (
    Calibration,
    CalibrationCompatibilityError,
    CalibrationError,
    calibrate_from_markers,
)
from backend.marker_detector import DEFAULT_DICT, MarkerDetection
from backend.models import (
    CalibrationFromLatestIn,
    CalibrationMeasureIn,
    CalibrationMeasureOut,
    CalibrationOut,
    CalibrationPreviewMarkerOut,
    CalibrationPreviewOut,
)

router = APIRouter()

LATEST_FRAME_MAX_AGE_SECONDS = 3.0
MAX_CALIBRATION_UPLOAD_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class _FrameSnapshot:
    frame: np.ndarray
    image_bytes: bytes | None
    content_type: str
    timestamp: float
    received_at: float
    age_seconds: float
    width: int
    height: int
    mode: str


def _to_out(calibration: Calibration) -> CalibrationOut:
    return CalibrationOut(
        camera_id=calibration.camera_id,
        marker_ids=calibration.marker_ids,
        width_m=calibration.width_m,
        height_m=calibration.height_m,
        homography=calibration.homography,
        source_frame_width=calibration.source_frame_width,
        source_frame_height=calibration.source_frame_height,
        source_aspect_ratio=calibration.source_aspect_ratio,
        marker_centers=calibration.marker_centers,
        marker_centers_px=calibration.marker_centers_px,
        quality=calibration.quality,
        calibration_quality=calibration.quality,
        created_at=calibration.created_at,
    )


def _parse_marker_ids(value: str) -> list[int]:
    try:
        marker_ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    except (AttributeError, ValueError) as exc:
        raise HTTPException(
            400, "marker_ids must be a comma-separated list of integers"
        ) from exc
    if len(marker_ids) != 4:
        raise HTTPException(400, "Need exactly 4 marker IDs (TL, TR, BR, BL)")
    if len(set(marker_ids)) != 4:
        raise HTTPException(400, "Marker IDs must be unique")
    invalid = [marker_id for marker_id in marker_ids if marker_id < 0 or marker_id >= 50]
    if invalid:
        raise HTTPException(
            400, f"Marker IDs outside DICT_4X4_50 range 0..49: {invalid}"
        )
    return marker_ids


def _invalid_quality(message: str) -> dict:
    return {
        "valid": False,
        "status": "invalid",
        "score": 0.0,
        "score_percent": 0.0,
        "coverage_ratio": 0.0,
        "quadrilateral_area_ratio": 0.0,
        "min_edge_ratio": 0.0,
        "max_edge_ratio": 0.0,
        "smallest_marker_area_px": 0.0,
        "min_angle_deg": 0.0,
        "max_angle_deg": 0.0,
        "reprojection_error_normalized": None,
        "reprojection_error_m": None,
        "condition_number": None,
        "warnings": [message],
        "messages": [message],
    }


def _decode_frame(image_bytes: bytes) -> np.ndarray | None:
    if not image_bytes:
        return None
    array = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _latest_item(request: Request, camera_id: str):
    store = getattr(request.app.state, "latest_frame_store", None)
    if store is None:
        return None
    # The current and next store contracts both expose get(camera_id).
    return store.get(camera_id)


def _snapshot_from_item(item) -> _FrameSnapshot:
    image_bytes = getattr(item, "image_bytes", None)
    if image_bytes is not None:
        image_bytes = bytes(image_bytes)

    stored_frame = getattr(item, "frame", None)
    if isinstance(stored_frame, np.ndarray) and stored_frame.size:
        frame = stored_frame.copy()
    else:
        frame = _decode_frame(image_bytes or b"")
    if frame is None or frame.size == 0:
        raise HTTPException(422, "Latest camera frame cannot be decoded")

    received_at = getattr(item, "received_at", None)
    if received_at is None or not math.isfinite(float(received_at)):
        raise HTTPException(409, "Latest camera frame has no valid receive timestamp")
    received_at = float(received_at)
    age_seconds = max(0.0, time.time() - received_at)
    timestamp = float(
        getattr(
            item,
            "timestamp",
            getattr(item, "captured_at", received_at),
        )
    )
    height, width = frame.shape[:2]
    return _FrameSnapshot(
        frame=frame,
        image_bytes=image_bytes,
        content_type=str(getattr(item, "content_type", "image/jpeg") or "image/jpeg"),
        timestamp=timestamp,
        received_at=received_at,
        age_seconds=age_seconds,
        width=int(width),
        height=int(height),
        mode=str(getattr(item, "mode", "site") or "site"),
    )


def _latest_snapshot(
    request: Request,
    camera_id: str,
    *,
    require_fresh: bool,
) -> _FrameSnapshot | None:
    item = _latest_item(request, camera_id)
    if item is None:
        if require_fresh:
            raise HTTPException(404, f"No frame available for camera {camera_id!r}")
        return None
    snapshot = _snapshot_from_item(item)
    if require_fresh and snapshot.age_seconds > LATEST_FRAME_MAX_AGE_SECONDS:
        raise HTTPException(
            409,
            "Latest frame is stale: "
            f"age {snapshot.age_seconds:.3f}s exceeds "
            f"{LATEST_FRAME_MAX_AGE_SECONDS:.1f}s",
        )
    return snapshot


def _calibrate_frame(
    request: Request,
    camera_id: str,
    frame: np.ndarray,
    marker_ids: list[int],
    width_m: float,
    height_m: float,
) -> Calibration:
    scheduler = getattr(request.app.state, "marker_scheduler", None)
    if scheduler is not None:
        # A submitted calibration must describe this exact frame, so it is
        # the one place where bypassing the 5 FPS preview throttle is useful.
        detections = scheduler.process(
            camera_id,
            frame,
            calibration=True,
            force=True,
        )
    else:
        detections = request.app.state.marker_detector.detect(frame)
    try:
        calibration = calibrate_from_markers(
            camera_id=camera_id,
            detections=detections,
            marker_ids=marker_ids,
            width_m=width_m,
            height_m=height_m,
            frame_width=int(frame.shape[1]),
            frame_height=int(frame.shape[0]),
        )
    except CalibrationError as exc:
        raise HTTPException(400, str(exc)) from exc
    request.app.state.calibration_store.set(calibration)
    return calibration


def _raw_frame_response(snapshot: _FrameSnapshot) -> Response:
    image_bytes = snapshot.image_bytes
    declared_type = snapshot.content_type.partition(";")[0].strip().lower()
    media_type = (
        declared_type
        if declared_type in {"image/jpeg", "image/png", "image/webp"}
        else "application/octet-stream"
    )
    if not image_bytes or media_type == "application/octet-stream":
        encoded, buffer = cv2.imencode(
            ".jpg",
            snapshot.frame,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        if not encoded:
            raise HTTPException(500, "Could not encode latest camera frame")
        image_bytes = buffer.tobytes()
        media_type = "image/jpeg"

    return Response(
        content=image_bytes,
        media_type=media_type,
        headers={
            "Cache-Control": "no-store",
            "X-Frame-Timestamp": str(snapshot.timestamp),
            "X-Frame-Captured-At": str(snapshot.timestamp),
            "X-Frame-Received-At": str(snapshot.received_at),
            "X-Frame-Age-Sec": f"{snapshot.age_seconds:.6f}",
            "X-Frame-Width": str(snapshot.width),
            "X-Frame-Height": str(snapshot.height),
        },
    )


@router.get("/calibration/markers/{marker_id}.png")
async def download_calibration_marker(marker_id: int):
    dictionary = cv2.aruco.getPredefinedDictionary(DEFAULT_DICT)
    marker_count = int(dictionary.bytesList.shape[0])
    if marker_id < 0 or marker_id >= marker_count:
        raise HTTPException(
            400,
            f"marker_id must be in DICT_4X4_50 range 0..{marker_count - 1}",
        )

    marker_size = 800
    margin = 120
    caption_height = 150
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(
            dictionary, marker_id, marker_size
        )
    else:  # pragma: no cover - OpenCV < 4.7
        marker = cv2.aruco.drawMarker(dictionary, marker_id, marker_size)

    canvas = np.full(
        (marker_size + margin * 2 + caption_height, marker_size + margin * 2),
        255,
        dtype=np.uint8,
    )
    canvas[
        margin:margin + marker_size,
        margin:margin + marker_size,
    ] = marker
    caption = f"ArUco DICT_4X4_50 - ID {marker_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.35
    thickness = 3
    (text_width, text_height), _baseline = cv2.getTextSize(
        caption, font, font_scale, thickness
    )
    text_x = max(20, (canvas.shape[1] - text_width) // 2)
    text_y = marker_size + margin + (caption_height + text_height) // 2
    cv2.putText(
        canvas,
        caption,
        (text_x, text_y),
        font,
        font_scale,
        0,
        thickness,
        cv2.LINE_AA,
    )
    encoded, buffer = cv2.imencode(
        ".png", canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    if not encoded:  # pragma: no cover
        raise HTTPException(500, "Could not generate marker PNG")
    filename = f"aruco_4x4_50_id_{marker_id}.png"
    return Response(
        content=buffer.tobytes(),
        media_type="image/png",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "public, max-age=86400",
        },
    )


@router.post("/calibration/{camera_id}", response_model=CalibrationOut)
async def create_calibration(
    request: Request,
    camera_id: str,
    image: UploadFile = File(...),
    marker_ids: str = Form(...),
    width_m: float = Form(...),
    height_m: float = Form(...),
):
    parsed_ids = _parse_marker_ids(marker_ids)
    raw = await image.read(MAX_CALIBRATION_UPLOAD_BYTES + 1)
    if len(raw) > MAX_CALIBRATION_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"Calibration image exceeds {MAX_CALIBRATION_UPLOAD_BYTES} bytes",
        )
    frame = _decode_frame(raw)
    if frame is None:
        raise HTTPException(400, "Could not decode calibration image")
    calibration = _calibrate_frame(
        request, camera_id, frame, parsed_ids, width_m, height_m
    )
    return _to_out(calibration)


@router.post(
    "/calibration/{camera_id}/from-latest",
    response_model=CalibrationOut,
)
async def create_calibration_from_latest(
    request: Request,
    camera_id: str,
    payload: CalibrationFromLatestIn,
):
    snapshot = _latest_snapshot(request, camera_id, require_fresh=True)
    assert snapshot is not None
    calibration = _calibrate_frame(
        request,
        camera_id,
        snapshot.frame,
        payload.marker_ids,
        payload.width_m,
        payload.height_m,
    )
    return _to_out(calibration)


@router.get(
    "/calibration/{camera_id}/preview",
    response_model=CalibrationPreviewOut,
)
async def calibration_preview(
    request: Request,
    camera_id: str,
    marker_ids: str = Query(default="10,20,30,40"),
):
    required_ids = _parse_marker_ids(marker_ids)
    scheduler = getattr(request.app.state, "marker_scheduler", None)
    if scheduler is not None:
        # Existing preview polling doubles as a short lease.  Closing the
        # calibration page therefore stops 5 FPS ArUco work automatically,
        # without a new public session API or fragile unload request.
        scheduler.touch_calibration_session(camera_id)
    snapshot = _latest_snapshot(request, camera_id, require_fresh=False)
    calibration = request.app.state.calibration_store.get(camera_id)
    calibration_out = _to_out(calibration) if calibration is not None else None

    if snapshot is None:
        error = "No frame available for this camera."
        return CalibrationPreviewOut(
            camera_id=camera_id,
            frame_available=False,
            fresh=False,
            required_marker_ids=required_ids,
            detected_markers=[],
            missing_marker_ids=required_ids,
            valid=False,
            validation_error=error,
            quality=_invalid_quality(error),
            calibration_quality=_invalid_quality(error),
            calibration_exists=calibration is not None,
            calibration_valid_for_frame=False,
            calibration_active=calibration is not None,
            calibration_aspect_compatible=False,
            calibration=calibration_out,
        )

    compatibility_warning = (
        calibration.compatibility_warning(snapshot.width, snapshot.height)
        if calibration is not None else None
    )
    calibration_compatible = calibration is not None and compatibility_warning is None
    is_fresh = snapshot.age_seconds <= LATEST_FRAME_MAX_AGE_SECONDS
    if not is_fresh:
        error = (
            "Latest frame is stale: "
            f"{snapshot.age_seconds:.3f}s > {LATEST_FRAME_MAX_AGE_SECONDS:.1f}s."
        )
        return CalibrationPreviewOut(
            camera_id=camera_id,
            frame_available=True,
            frame_age_sec=round(snapshot.age_seconds, 4),
            timestamp=snapshot.timestamp,
            received_at=snapshot.received_at,
            mode=snapshot.mode,
            frame_width=snapshot.width,
            frame_height=snapshot.height,
            frame_aspect=snapshot.width / snapshot.height,
            fresh=False,
            required_marker_ids=required_ids,
            detected_markers=[],
            missing_marker_ids=required_ids,
            valid=False,
            validation_error=error,
            quality=_invalid_quality(error),
            calibration_quality=_invalid_quality(error),
            calibration_exists=calibration is not None,
            calibration_valid_for_frame=calibration_compatible,
            calibration_active=calibration is not None,
            calibration_aspect_compatible=calibration_compatible,
            calibration_compatibility_warning=compatibility_warning,
            calibration=calibration_out,
        )

    if scheduler is not None:
        detections: list[MarkerDetection] = scheduler.process(
            camera_id,
            snapshot.frame,
            calibration=True,
        )
    else:
        detections = request.app.state.marker_detector.detect(snapshot.frame)
    detected_markers = [
        CalibrationPreviewMarkerOut(
            marker_id=detection.marker_id,
            center=[float(value) for value in detection.center],
            center_normalized=[
                float(detection.center[0]) / snapshot.width,
                float(detection.center[1]) / snapshot.height,
            ],
            corners=[
                [float(x), float(y)] for x, y in detection.corners
            ],
        )
        for detection in sorted(detections, key=lambda item: item.marker_id)
    ]
    detected_ids = {detection.marker_id for detection in detections}
    missing_ids = [
        marker_id for marker_id in required_ids if marker_id not in detected_ids
    ]
    quality = None
    validation_error = None
    valid = False
    if missing_ids:
        validation_error = f"Markers not detected in frame: {missing_ids}"
        quality = _invalid_quality(validation_error)
    else:
        try:
            probe = calibrate_from_markers(
                camera_id=camera_id,
                detections=detections,
                marker_ids=required_ids,
                width_m=1.0,
                height_m=1.0,
                frame_width=snapshot.width,
                frame_height=snapshot.height,
            )
            quality = probe.quality
            valid = True
        except CalibrationError as exc:
            validation_error = str(exc)
            quality = _invalid_quality(validation_error)

    return CalibrationPreviewOut(
        camera_id=camera_id,
        frame_available=True,
        frame_age_sec=round(snapshot.age_seconds, 4),
        timestamp=snapshot.timestamp,
        received_at=snapshot.received_at,
        mode=snapshot.mode,
        frame_width=snapshot.width,
        frame_height=snapshot.height,
        frame_aspect=snapshot.width / snapshot.height,
        fresh=True,
        required_marker_ids=required_ids,
        detected_markers=detected_markers,
        missing_marker_ids=missing_ids,
        valid=valid,
        validation_error=validation_error,
        quality=quality,
        calibration_quality=quality,
        calibration_exists=calibration is not None,
        calibration_valid_for_frame=calibration_compatible,
        calibration_active=calibration is not None,
        calibration_aspect_compatible=calibration_compatible,
        calibration_compatibility_warning=compatibility_warning,
        calibration=calibration_out,
    )


@router.get("/calibration/{camera_id}/latest-frame")
async def get_latest_calibration_frame(request: Request, camera_id: str):
    """Backward-compatible raw preview for clients that still need it."""
    snapshot = _latest_snapshot(request, camera_id, require_fresh=True)
    assert snapshot is not None
    return _raw_frame_response(snapshot)


@router.post(
    "/calibration/{camera_id}/measure",
    response_model=CalibrationMeasureOut,
)
async def measure_calibrated_distance(
    request: Request,
    camera_id: str,
    payload: CalibrationMeasureIn,
):
    calibration = request.app.state.calibration_store.get(camera_id)
    if calibration is None:
        raise HTTPException(404, f"No calibration for camera {camera_id!r}")
    warning = calibration.compatibility_warning(
        payload.frame_width, payload.frame_height
    )
    if warning:
        raise HTTPException(409, warning)

    for name, point in (("point_a", payload.point_a), ("point_b", payload.point_b)):
        if (
            point[0] < 0.0
            or point[0] > payload.frame_width
            or point[1] < 0.0
            or point[1] > payload.frame_height
        ):
            raise HTTPException(
                400,
                f"{name} must lie inside frame "
                f"0..{payload.frame_width} x 0..{payload.frame_height}",
            )
    try:
        point_a_m = calibration.project(
            payload.point_a[0],
            payload.point_a[1],
            payload.frame_width,
            payload.frame_height,
        )
        point_b_m = calibration.project(
            payload.point_b[0],
            payload.point_b[1],
            payload.frame_width,
            payload.frame_height,
        )
    except CalibrationCompatibilityError as exc:
        raise HTTPException(409, str(exc)) from exc
    except CalibrationError as exc:
        raise HTTPException(400, str(exc)) from exc
    distance_m = math.dist(point_a_m, point_b_m)
    return CalibrationMeasureOut(
        point_a_m=[float(point_a_m[0]), float(point_a_m[1])],
        point_b_m=[float(point_b_m[0]), float(point_b_m[1])],
        distance_m=float(distance_m),
    )


@router.get("/calibration/{camera_id}", response_model=CalibrationOut)
async def get_calibration(request: Request, camera_id: str):
    calibration = request.app.state.calibration_store.get(camera_id)
    if calibration is None:
        raise HTTPException(404, "No calibration for this camera")
    return _to_out(calibration)


@router.delete("/calibration/{camera_id}")
async def delete_calibration(request: Request, camera_id: str):
    removed = request.app.state.calibration_store.clear(camera_id)
    if not removed:
        raise HTTPException(404, "No calibration for this camera")
    return {"status": "deleted", "camera_id": camera_id}
