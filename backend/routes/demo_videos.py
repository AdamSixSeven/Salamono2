"""HTTP control plane for backend-decoded demonstration videos."""
from __future__ import annotations

import asyncio
from typing import Callable

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse

from backend.demo_video import (
    DemoVideoConflictError,
    DemoVideoError,
    DemoVideoJobSnapshot,
    DemoVideoNotFoundError,
    DemoVideoProcessingDisabledError,
    DemoVideoService,
    DemoVideoTimeoutError,
    DemoVideoTooLargeError,
    DemoVideoValidationError,
)


router = APIRouter(prefix="/demo-videos", tags=["demo-videos"])


def _service(request: Request) -> DemoVideoService:
    service = getattr(request.app.state, "demo_video_service", None)
    if service is None:
        raise HTTPException(503, "Demo video service is not initialized")
    return service


def _public_snapshot(snapshot: DemoVideoJobSnapshot) -> dict:
    data = snapshot.to_dict()
    # Keep server filesystem layout internal; clients only need the stable job
    # identifier and playback metadata.
    data.pop("input_path", None)
    data["output_url"] = (
        f"/api/demo-videos/{snapshot.job_id}/output"
        if data.get("export_status") == "ready"
        else None
    )
    return data


def _raise_http(exc: BaseException) -> None:
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


async def _control(
    operation: Callable[..., DemoVideoJobSnapshot],
    job_id: str,
) -> dict:
    try:
        snapshot = await asyncio.to_thread(operation, job_id)
    except DemoVideoError as exc:
        _raise_http(exc)
        raise AssertionError("unreachable")
    return _public_snapshot(snapshot)


@router.post("")
async def upload_demo_video(
    request: Request,
    video: UploadFile = File(...),
    camera_id: str = Form(default="demo_upload"),
    mode: str = Form(default="site"),
    playback_mode: str | None = Form(default=None),
):
    """Persist one multipart upload, then probe it on a background worker."""

    service = _service(request)
    if not bool(service.config.processing_enabled):
        _raise_http(DemoVideoProcessingDisabledError(
            "Demo video processing is disabled"
        ))

    job_id: str | None = None
    write_task: asyncio.Task | None = None
    try:
        created = await asyncio.to_thread(
            service.create_upload,
            video.filename or "",
            video.content_type or "",
            camera_id=camera_id,
            mode=mode,
            playback_mode=playback_mode,
        )
        job_id = created.job_id
        await video.seek(0)
        write_task = asyncio.create_task(
            asyncio.to_thread(
                service.write_upload,
                job_id,
                video.file,
            )
        )
        # Keep the local streaming copy alive if the browser closes the HTTP
        # request; the cancellation branch waits for it and removes the job.
        # This prevents an aborted upload from leaving an untracked directory.
        uploaded = await asyncio.shield(write_task)
        # The response intentionally retains the deterministic `uploaded`
        # snapshot.  The independent probe has already been scheduled, so the
        # status endpoint will next expose `loading`, then `ready` or `failed`.
        await asyncio.to_thread(service.start_probe, job_id)
        return _public_snapshot(uploaded)
    except asyncio.CancelledError:
        if write_task is not None and not write_task.done():
            try:
                await asyncio.shield(write_task)
            except Exception:
                pass
        if job_id is not None:
            try:
                await asyncio.to_thread(service.delete, job_id)
            except DemoVideoError:
                pass
        raise
    except DemoVideoError as exc:
        if job_id is not None:
            try:
                await asyncio.to_thread(service.delete, job_id)
            except DemoVideoError:
                pass
        _raise_http(exc)
        raise AssertionError("unreachable")
    except Exception as exc:
        if job_id is not None:
            try:
                await asyncio.to_thread(service.delete, job_id)
            except DemoVideoError:
                pass
        raise HTTPException(500, "Demo video upload could not be stored") from exc
    finally:
        await video.close()


@router.get("/{job_id}")
async def get_demo_video(request: Request, job_id: str):
    service = _service(request)
    try:
        snapshot = await asyncio.to_thread(service.get, job_id)
    except DemoVideoError as exc:
        _raise_http(exc)
        raise AssertionError("unreachable")
    data = _public_snapshot(snapshot)
    data["metrics"] = await asyncio.to_thread(service.stats)
    return data


@router.post("/{job_id}/play")
async def play_demo_video(request: Request, job_id: str):
    service = _service(request)
    return await _control(service.play, job_id)


@router.post("/{job_id}/pause")
async def pause_demo_video(request: Request, job_id: str):
    service = _service(request)
    return await _control(service.pause, job_id)


@router.post("/{job_id}/resume")
async def resume_demo_video(request: Request, job_id: str):
    service = _service(request)
    return await _control(service.resume, job_id)


@router.post("/{job_id}/restart")
async def restart_demo_video(request: Request, job_id: str):
    service = _service(request)
    return await _control(service.restart, job_id)


@router.post("/{job_id}/stop")
async def stop_demo_video(request: Request, job_id: str):
    service = _service(request)
    return await _control(service.stop, job_id)


@router.post("/{job_id}/export")
async def export_demo_video(request: Request, job_id: str):
    service = _service(request)
    return await _control(service.export, job_id)


@router.get("/{job_id}/output")
async def download_demo_video_output(request: Request, job_id: str):
    service = _service(request)
    try:
        path = await asyncio.to_thread(service.output_path_for, job_id)
    except DemoVideoError as exc:
        _raise_http(exc)
        raise AssertionError("unreachable")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@router.delete("/{job_id}", status_code=204)
async def delete_demo_video(request: Request, job_id: str) -> Response:
    service = _service(request)
    try:
        await asyncio.to_thread(service.delete, job_id)
    except DemoVideoError as exc:
        _raise_http(exc)
    return Response(status_code=204)
