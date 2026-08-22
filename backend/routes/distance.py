"""Simple ground-plane distance preview using the existing Perimetr detector."""
from __future__ import annotations

import asyncio
import base64
from contextlib import nullcontext
from dataclasses import dataclass
import math
import threading
import time
import uuid

import cv2
from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.demo_video import (
    DemoVideoConflictError,
    DemoVideoError,
    DemoVideoInputLease,
    DemoVideoJobSnapshot,
    DemoVideoNotFoundError,
    DemoVideoProcessingDisabledError,
    DemoVideoService,
    DemoVideoTimeoutError,
    DemoVideoTooLargeError,
    DemoVideoValidationError,
)
from backend.distance_measurement import DistancePair, GroundPlaneDistanceService
from config import CONFIG


router = APIRouter(prefix="/distance", tags=["distance"])
_service = GroundPlaneDistanceService()
_DISTANCE_VIDEO_CAMERA_PREFIX = "distance_clip:"
_HAZARD_CATEGORIES = frozenset({"vehicle", "hazard", "danger"})
_MANUAL_SLOT_INIT_LOCK = threading.Lock()


class _LeasedFileResponse(FileResponse):
    """Keep a source lease through normal, Range and disconnected responses."""

    def __init__(self, lease: DemoVideoInputLease, **kwargs):
        self._source_lease = lease
        super().__init__(path=lease.path, **kwargs)

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._source_lease.release()


class VideoFrameIn(BaseModel):
    time_sec: float | None = Field(default=None, ge=0.0)
    frame_index: int | None = Field(default=None, ge=0)
    max_width: int = Field(default=1280, ge=320, le=1920)


class VideoAnalyzeIn(BaseModel):
    time_sec: float | None = Field(default=None, ge=0.0)
    frame_index: int | None = Field(default=None, ge=0)
    max_distance_m: float | None = Field(default=None, gt=0.0, le=500.0)
    preview_max_width: int = Field(default=1280, ge=320, le=1920)


@dataclass(frozen=True, slots=True)
class _DecodedVideoFrame:
    frame: object
    requested_time_sec: float | None
    requested_frame_index: int | None
    target_frame_index: int | None
    actual_time_sec: float
    frame_index: int
    fps: float | None
    frame_count: int | None
    width: int
    height: int
    seek_mode: str


def _scale_box(box, scale: float) -> tuple[int, int, int, int]:
    return tuple(int(round(float(value) * scale)) for value in box)


def _scale_point(point, scale: float) -> tuple[int, int]:
    return int(round(point[0] * scale)), int(round(point[1] * scale))


def _distance_color(distance_m: float) -> tuple[int, int, int]:
    if distance_m <= float(CONFIG.danger.danger_distance_m):
        return (40, 40, 230)
    if distance_m <= float(CONFIG.danger.warning_distance_m):
        return (0, 180, 255)
    return (70, 190, 70)


def _distance_level(distance_m: float) -> str:
    if distance_m <= float(CONFIG.danger.danger_distance_m):
        return "danger"
    if distance_m <= float(CONFIG.danger.warning_distance_m):
        return "warning"
    return "safe"


def _render_preview(frame, detections, pairs: list[DistancePair], max_width: int):
    height, width = frame.shape[:2]
    scale = min(1.0, max(320, int(max_width)) / max(1, width))
    if scale < 1.0:
        preview = cv2.resize(
            frame,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        preview = frame.copy()

    thickness = max(1, int(round(2 * max(0.7, scale))))
    font_scale = max(0.45, 0.62 * max(0.8, scale))

    for detection in detections:
        if detection.category not in {"person", "vehicle", "hazard", "danger"}:
            continue
        x1, y1, x2, y2 = _scale_box(detection.box, scale)
        is_person = detection.category == "person"
        color = (60, 205, 80) if is_person else (0, 145, 255)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, thickness)
        label = f"{detection.class_name} {detection.confidence * 100:.0f}%"
        cv2.putText(
            preview,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, thickness),
            cv2.LINE_AA,
        )

    # Draw closest relationships last so labels stay readable.
    for pair in reversed(pairs[:64]):
        a = _scale_point(pair.person_point_px, scale)
        b = _scale_point(pair.hazard_point_px, scale)
        color = _distance_color(pair.distance_m)
        cv2.line(preview, a, b, color, max(1, thickness), cv2.LINE_AA)
        cx, cy = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
        text = f"{pair.distance_m:.2f} m"
        cv2.putText(
            preview,
            text,
            (cx + 5, max(18, cy - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, thickness),
            cv2.LINE_AA,
        )
    return preview


def _pair_json(pair: DistancePair) -> dict:
    return {
        "person_index": pair.person_index,
        "hazard_index": pair.hazard_index,
        "person_class": pair.person_class,
        "hazard_class": pair.hazard_class,
        "person_confidence": round(pair.person_confidence, 6),
        "hazard_confidence": round(pair.hazard_confidence, 6),
        "person_box": list(pair.person_box),
        "hazard_box": list(pair.hazard_box),
        "distance_m": round(pair.distance_m, 4),
        "status": _distance_level(pair.distance_m),
        "person_point_px": list(pair.person_point_px),
        "hazard_point_px": list(pair.hazard_point_px),
    }


def _video_service(request: Request) -> DemoVideoService:
    service = getattr(request.app.state, "demo_video_service", None)
    if service is None:
        raise HTTPException(503, "Magazyn klipów wideo nie jest gotowy.")
    return service


def _manual_slot(request: Request) -> threading.BoundedSemaphore:
    slot = getattr(request.app.state, "distance_video_manual_slot", None)
    if slot is not None:
        return slot
    with _MANUAL_SLOT_INIT_LOCK:
        slot = getattr(request.app.state, "distance_video_manual_slot", None)
        if slot is None:
            slot = threading.BoundedSemaphore(1)
            request.app.state.distance_video_manual_slot = slot
        return slot


def _run_manual_operation(slot, operation, *args):
    try:
        return operation(*args)
    finally:
        # This runs in the worker thread.  If the HTTP coroutine is cancelled,
        # asyncio.to_thread keeps running and the slot remains held until raw
        # decode/inference really finishes.
        slot.release()


async def _submit_manual_operation(request: Request, operation, *args):
    slot = _manual_slot(request)
    if not slot.acquire(blocking=False):
        raise HTTPException(
            429,
            "Trwa już analiza ręcznej klatki. Spróbuj ponownie za chwilę.",
            headers={"Retry-After": "1"},
        )
    try:
        worker = asyncio.create_task(
            asyncio.to_thread(_run_manual_operation, slot, operation, *args)
        )
    except BaseException:
        slot.release()
        raise
    return await asyncio.shield(worker)


def _raise_video_http(exc: BaseException) -> None:
    if isinstance(exc, DemoVideoNotFoundError):
        raise HTTPException(404, str(exc)) from exc
    if isinstance(exc, DemoVideoTooLargeError):
        raise HTTPException(413, str(exc)) from exc
    if isinstance(exc, DemoVideoValidationError):
        raise HTTPException(415, str(exc)) from exc
    if isinstance(exc, DemoVideoConflictError):
        raise HTTPException(409, str(exc)) from exc
    if isinstance(exc, DemoVideoTimeoutError):
        raise HTTPException(504, str(exc)) from exc
    if isinstance(exc, DemoVideoProcessingDisabledError):
        raise HTTPException(503, str(exc)) from exc
    if isinstance(exc, DemoVideoError):
        raise HTTPException(500, str(exc)) from exc
    raise exc


def _distance_video_snapshot(
    service: DemoVideoService,
    video_id: str,
    *,
    require_ready: bool = False,
) -> DemoVideoJobSnapshot:
    try:
        snapshot = service.get(video_id)
    except DemoVideoError as exc:
        _raise_video_http(exc)
        raise AssertionError("unreachable")
    try:
        asset_kind = service.asset_kind_for(video_id)
    except DemoVideoError as exc:
        _raise_video_http(exc)
        raise AssertionError("unreachable")
    if asset_kind != "distance":
        raise HTTPException(404, f"Unknown distance video: {video_id}")
    if require_ready and snapshot.status != "ready":
        if snapshot.status == "failed":
            raise HTTPException(415, snapshot.error or "Nie można odczytać klipu wideo.")
        raise HTTPException(409, f"Klip nie jest gotowy (status: {snapshot.status}).")
    return snapshot


def _metadata_json(snapshot: DemoVideoJobSnapshot) -> dict:
    calibration = (
        _service.status(snapshot.width, snapshot.height)
        if snapshot.width is not None and snapshot.height is not None
        else _service.status()
    )
    return {
        "id": snapshot.job_id,
        "job_id": snapshot.job_id,
        "filename": snapshot.filename,
        "mime_type": snapshot.mime_type,
        "size_bytes": snapshot.size_bytes,
        "width": snapshot.width,
        "height": snapshot.height,
        "fps": snapshot.source_fps,
        "source_fps": snapshot.source_fps,
        "frame_count": snapshot.total_frames,
        "total_frames": snapshot.total_frames,
        "duration_sec": snapshot.duration_sec,
        "status": snapshot.status,
        "error": snapshot.error,
        "video_url": f"/api/distance/videos/{snapshot.job_id}/content",
        "calibration": calibration,
    }


def _positive_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def _capture_value(capture, property_id: int) -> float | None:
    try:
        value = float(capture.get(property_id))
    except (AttributeError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _extract_video_frame(
    service: DemoVideoService,
    snapshot: DemoVideoJobSnapshot,
    *,
    time_sec: float | None,
    frame_index: int | None,
) -> _DecodedVideoFrame:
    if time_sec is not None and not math.isfinite(float(time_sec)):
        raise DemoVideoValidationError("time_sec must be finite")

    capture = service.open_input_capture(snapshot.job_id)
    try:
        if not bool(capture.isOpened()):
            raise DemoVideoValidationError("Video cannot be opened")

        fps = _positive_number(snapshot.source_fps) or _positive_number(
            _capture_value(capture, cv2.CAP_PROP_FPS)
        )
        capture_count = _positive_number(
            _capture_value(capture, cv2.CAP_PROP_FRAME_COUNT)
        )
        frame_count = snapshot.total_frames
        if frame_count is None and capture_count is not None:
            frame_count = max(1, int(round(capture_count)))

        requested_time = float(time_sec) if time_sec is not None else None
        requested_index = int(frame_index) if frame_index is not None else None
        seek_mode = "frame" if requested_index is not None else "time"
        if requested_index is None and requested_time is None:
            requested_time = 0.0

        target_index: int | None
        if seek_mode == "frame":
            assert requested_index is not None
            target_index = requested_index
            if requested_time is None and fps is not None:
                requested_time = float(requested_index) / fps
        elif fps is not None:
            assert requested_time is not None
            target_index = int(math.floor(requested_time * fps + 0.5))
            requested_index = target_index
        else:
            target_index = None

        if target_index is not None and frame_count is not None:
            target_index = min(target_index, max(0, int(frame_count) - 1))

        setter = getattr(capture, "set", None)
        if not callable(setter):
            if target_index not in {None, 0} or (requested_time or 0.0) > 0.0:
                raise DemoVideoValidationError("Video decoder does not support seeking")
        elif seek_mode == "frame":
            assert target_index is not None
            positioned = setter(cv2.CAP_PROP_POS_FRAMES, float(target_index))
            if positioned is False:
                raise DemoVideoValidationError("Video decoder could not seek to the frame")
        else:
            assert requested_time is not None
            positioned = setter(cv2.CAP_PROP_POS_MSEC, requested_time * 1000.0)
            if positioned is False:
                raise DemoVideoValidationError("Video decoder could not seek to the timestamp")

        ok, frame = capture.read()
        if not ok or frame is None or getattr(frame, "size", 0) <= 0:
            raise DemoVideoValidationError("Requested video frame could not be decoded")
        if len(frame.shape) < 2:
            raise DemoVideoValidationError("Decoded video frame has an invalid shape")
        height, width = (int(frame.shape[0]), int(frame.shape[1]))

        next_position = _capture_value(capture, cv2.CAP_PROP_POS_FRAMES)
        if next_position is not None and next_position >= 1.0:
            actual_index = max(0, int(round(next_position)) - 1)
        elif target_index is not None:
            actual_index = target_index
        else:
            actual_index = 0

        position_msec = _capture_value(capture, cv2.CAP_PROP_POS_MSEC)
        if position_msec is not None and (position_msec > 0.0 or actual_index == 0):
            actual_time = max(0.0, position_msec / 1000.0)
        elif fps is not None:
            actual_time = float(actual_index) / fps
        else:
            actual_time = float(requested_time or 0.0)

        return _DecodedVideoFrame(
            frame=frame,
            requested_time_sec=requested_time,
            requested_frame_index=requested_index,
            target_frame_index=target_index,
            actual_time_sec=actual_time,
            frame_index=actual_index,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            seek_mode=seek_mode,
        )
    finally:
        capture.release()


def _encode_preview(frame, detections, pairs, max_width: int) -> dict:
    preview = _render_preview(frame, detections, pairs, max_width=max_width)
    ok, encoded = cv2.imencode(
        ".jpg",
        preview,
        [int(cv2.IMWRITE_JPEG_QUALITY), 82],
    )
    if not ok:
        raise RuntimeError("Nie udało się zakodować podglądu JPEG.")
    preview_height, preview_width = preview.shape[:2]
    source_width = max(1, int(frame.shape[1]))
    return {
        "preview_width": int(preview_width),
        "preview_height": int(preview_height),
        "preview_scale": round(float(preview_width) / source_width, 8),
        "preview_jpeg_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


def _frame_json(decoded: _DecodedVideoFrame) -> dict:
    return {
        "requested_time_sec": decoded.requested_time_sec,
        "requested_frame_index": decoded.requested_frame_index,
        "target_frame_index": decoded.target_frame_index,
        "actual_time_sec": round(decoded.actual_time_sec, 6),
        "frame_index": decoded.frame_index,
        "fps": decoded.fps,
        "frame_count": decoded.frame_count,
        "width": decoded.width,
        "height": decoded.height,
        "seek_mode": decoded.seek_mode,
    }


def _preview_video_frame(
    service: DemoVideoService,
    snapshot: DemoVideoJobSnapshot,
    payload: VideoFrameIn,
) -> dict:
    decoded = _extract_video_frame(
        service,
        snapshot,
        time_sec=payload.time_sec,
        frame_index=payload.frame_index,
    )
    response = {"id": snapshot.job_id, **_frame_json(decoded)}
    response.update(
        _encode_preview(
            decoded.frame,
            [],
            [],
            payload.max_width,
        )
    )
    return response


def _detection_json(detection, index: int) -> dict:
    return {
        "index": index,
        "class_id": int(detection.class_id),
        "class_name": str(detection.class_name),
        "category": str(detection.category),
        "confidence": round(float(detection.confidence), 6),
        "box": [int(value) for value in detection.box],
        "track_id": detection.track_id,
    }


def _overall_distance_status(pairs: list[DistancePair]) -> str:
    if not pairs:
        return "no_pairs"
    closest = min(pair.distance_m for pair in pairs)
    return _distance_level(closest)


def _analyze_video_frame(
    request: Request,
    service: DemoVideoService,
    snapshot: DemoVideoJobSnapshot,
    payload: VideoAnalyzeIn,
) -> dict:
    started = time.monotonic()
    decoded = _extract_video_frame(
        service,
        snapshot,
        time_sec=payload.time_sec,
        frame_index=payload.frame_index,
    )

    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise RuntimeError("Perimetr SceneDetector is not initialized")
    analysis_lock = getattr(request.app.state, "analysis_lock", None)
    lock_context = analysis_lock if analysis_lock is not None else nullcontext()
    with lock_context:
        # This is the same long-lived SceneDetector instance used by live/demo
        # processing.  Only one selected source frame enters inference.
        detections = list(detector.detect(decoded.frame))

    person_count = sum(item.category == "person" for item in detections)
    hazard_count = sum(item.category in _HAZARD_CATEGORIES for item in detections)
    calibration = _service.status(decoded.width, decoded.height)
    distance_available = bool(calibration.get("available", _service.available))
    pairs: list[DistancePair] = []
    message = None
    if not distance_available:
        status = "calibration_unavailable"
        message = calibration.get("error") or "Brak kalibracji modułu dystansu."
    elif calibration.get("compatible") is False:
        status = "calibration_incompatible"
        message = calibration.get("warning") or (
            "Aspect ratio klipu jest niezgodny z kalibracją dystansu."
        )
    else:
        pairs, _measured_persons, _measured_hazards = _service.measure(
            detections,
            decoded.width,
            decoded.height,
            max_distance_m=payload.max_distance_m,
        )
        status = _overall_distance_status(pairs)
        if not pairs:
            message = "Nie wykryto pary osoba–maszyna w wybranej klatce."

    response = {
        "id": snapshot.job_id,
        **_frame_json(decoded),
        "status": status,
        "analysis_status": status,
        "message": message,
        "calibration": calibration,
        "person_count": person_count,
        "hazard_count": hazard_count,
        "pair_count": len(pairs),
        "detections": [
            _detection_json(detection, index)
            for index, detection in enumerate(detections)
        ],
        "pairs": [_pair_json(pair) for pair in pairs],
        "analysis_ms": round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
    }
    response.update(
        _encode_preview(
            decoded.frame,
            detections,
            pairs,
            max_width=payload.preview_max_width,
        )
    )
    return response


@router.post("/videos")
async def upload_distance_video(
    request: Request,
    video: UploadFile = File(...),
):
    """Store a clip cheaply and decode only one frame for metadata."""

    service = _video_service(request)
    video_id: str | None = None
    write_task: asyncio.Task | None = None
    try:
        created = await asyncio.to_thread(
            service.create_upload,
            video.filename or "",
            video.content_type or "",
            camera_id=f"{_DISTANCE_VIDEO_CAMERA_PREFIX}{uuid.uuid4().hex}",
            mode="site",
            playback_mode="fast",
            asset_kind="distance",
        )
        video_id = created.job_id
        await video.seek(0)
        write_task = asyncio.create_task(
            asyncio.to_thread(service.write_upload, video_id, video.file)
        )
        await asyncio.shield(write_task)
        # This read-only metadata probe bypasses playback worker capacity.  It
        # opens an independent capture and never runs the detector/pipeline.
        await asyncio.to_thread(service.start_read_only_probe, video_id)
        snapshot = await asyncio.to_thread(
            service.wait_for_status,
            video_id,
            {"ready", "failed"},
            timeout=max(1.0, float(service.config.probe_timeout_seconds) + 1.0),
        )
        if snapshot.status != "ready":
            raise DemoVideoValidationError(
                snapshot.error or "Nie można odczytać metadanych klipu wideo."
            )
        return _metadata_json(snapshot)
    except asyncio.CancelledError:
        if write_task is not None and not write_task.done():
            try:
                await asyncio.shield(write_task)
            except Exception:
                pass
        if video_id is not None:
            try:
                await asyncio.to_thread(service.delete, video_id)
            except DemoVideoError:
                pass
        raise
    except DemoVideoError as exc:
        if video_id is not None:
            try:
                await asyncio.to_thread(service.delete, video_id)
            except DemoVideoError:
                pass
        _raise_video_http(exc)
        raise AssertionError("unreachable")
    except Exception as exc:
        if video_id is not None:
            try:
                await asyncio.to_thread(service.delete, video_id)
            except DemoVideoError:
                pass
        raise HTTPException(500, "Nie udało się zapisać klipu wideo.") from exc
    finally:
        await video.close()


@router.get("/videos/{video_id}")
async def get_distance_video(request: Request, video_id: str):
    service = _video_service(request)
    snapshot = await asyncio.to_thread(
        _distance_video_snapshot,
        service,
        video_id,
    )
    return _metadata_json(snapshot)


@router.get("/videos/{video_id}/content")
async def get_distance_video_content(request: Request, video_id: str):
    service = _video_service(request)
    snapshot = await asyncio.to_thread(
        _distance_video_snapshot,
        service,
        video_id,
        require_ready=True,
    )
    lease = None
    try:
        lease = await asyncio.to_thread(service.acquire_input_lease, video_id)
    except DemoVideoError as exc:
        _raise_video_http(exc)
        raise AssertionError("unreachable")
    try:
        return _LeasedFileResponse(
            lease,
            media_type=snapshot.mime_type or "application/octet-stream",
        )
    except BaseException:
        lease.release()
        raise


@router.post("/videos/{video_id}/frame")
async def get_distance_video_frame(
    request: Request,
    video_id: str,
    payload: VideoFrameIn,
):
    service = _video_service(request)
    snapshot = await asyncio.to_thread(
        _distance_video_snapshot,
        service,
        video_id,
        require_ready=True,
    )
    try:
        return await _submit_manual_operation(
            request,
            _preview_video_frame,
            service,
            snapshot,
            payload,
        )
    except DemoVideoError as exc:
        _raise_video_http(exc)
        raise AssertionError("unreachable")


@router.post("/videos/{video_id}/analyze")
async def analyze_distance_video_frame(
    request: Request,
    video_id: str,
    payload: VideoAnalyzeIn,
):
    service = _video_service(request)
    snapshot = await asyncio.to_thread(
        _distance_video_snapshot,
        service,
        video_id,
        require_ready=True,
    )
    if getattr(request.app.state, "detector", None) is None:
        raise HTTPException(503, "Perimetr SceneDetector nie jest gotowy.")
    try:
        return await _submit_manual_operation(
            request,
            _analyze_video_frame,
            request,
            service,
            snapshot,
            payload,
        )
    except DemoVideoError as exc:
        _raise_video_http(exc)
        raise AssertionError("unreachable")


@router.delete("/videos/{video_id}", status_code=204)
async def delete_distance_video(request: Request, video_id: str) -> Response:
    service = _video_service(request)
    await asyncio.to_thread(_distance_video_snapshot, service, video_id)
    try:
        await asyncio.to_thread(service.delete, video_id)
    except DemoVideoError as exc:
        _raise_video_http(exc)
    return Response(status_code=204)


@router.get("/status")
def distance_status():
    return _service.status()


@router.get("/latest")
def distance_latest(
    request: Request,
    camera_id: str = Query("cam_default", min_length=1, max_length=128),
    max_width: int = Query(1280, ge=320, le=1920),
    max_distance_m: float | None = Query(None, gt=0.0, le=500.0),
):
    latest_store = getattr(request.app.state, "latest_frame_store", None)
    detection_store = getattr(request.app.state, "latest_detection_store", None)
    if latest_store is None or detection_store is None:
        return {
            "available": False,
            "camera_id": camera_id,
            "message": "Magazyn klatek/detekcji nie jest gotowy.",
            "calibration": _service.status(),
            "pairs": [],
        }

    latest = latest_store.get(camera_id, max_age_seconds=15.0)
    if latest is None:
        return {
            "available": _service.available,
            "camera_id": camera_id,
            "frame_available": False,
            "message": "Brak świeżej klatki dla tej kamery.",
            "calibration": _service.status(),
            "pairs": [],
        }

    calibration = _service.status(latest.width, latest.height)
    snapshot = detection_store.get_exact(
        camera_id,
        latest.timestamp,
        latest.width,
        latest.height,
        timestamp_tolerance=1e-4,
    )
    detections = list(snapshot.detections) if snapshot is not None else []
    pairs, person_count, hazard_count = _service.measure(
        detections,
        latest.width,
        latest.height,
        max_distance_m=max_distance_m,
    )

    preview = _render_preview(latest.frame, detections, pairs, max_width=max_width)
    ok, encoded = cv2.imencode(
        ".jpg",
        preview,
        [int(cv2.IMWRITE_JPEG_QUALITY), 82],
    )
    preview_b64 = base64.b64encode(encoded.tobytes()).decode("ascii") if ok else None

    message = None
    if not _service.available:
        message = _service.error or "Brak kalibracji modułu dystansu."
    elif calibration.get("compatible") is False:
        message = calibration.get("warning")
    elif snapshot is None:
        message = "Klatka jest dostępna, ale czekam na detekcję YOLO z tej samej klatki."

    return {
        "available": _service.available,
        "camera_id": camera_id,
        "frame_available": True,
        "detection_exact": snapshot is not None,
        "frame_timestamp": latest.timestamp,
        "frame_age_sec": round(max(0.0, time.time() - latest.received_at), 3),
        "frame_width": latest.width,
        "frame_height": latest.height,
        "person_count": person_count,
        "hazard_count": hazard_count,
        "pair_count": len(pairs),
        "message": message,
        "calibration": calibration,
        "pairs": [_pair_json(pair) for pair in pairs],
        "preview_jpeg_b64": preview_b64,
    }
