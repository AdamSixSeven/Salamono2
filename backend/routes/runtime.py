"""API for dynamically enabling/disabling expensive processing per camera."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/runtime", tags=["runtime"])


class RuntimeOptionsPatch(BaseModel):
    boxes: bool | None = None
    posture: bool | None = None
    zones: bool | None = None
    markers: bool | None = None
    distances: bool | None = None
    worker_id: bool | None = None
    ppe: bool | None = None


def _store(request: Request):
    store = getattr(request.app.state, "runtime_processing_store", None)
    if store is None:
        raise HTTPException(503, "Runtime processing store is unavailable.")
    return store


def _payload(options, mode: str = "site") -> dict:
    return {
        "options": options.to_dict(),
        "enabled_modules": options.enabled_modules(),
        "detector_required": options.requires_detector(mode),
        "mode": mode,
    }


@router.get("/{camera_id}")
async def get_runtime_options(camera_id: str, request: Request, mode: str = "site"):
    return _payload(_store(request).get(camera_id), mode)


@router.patch("/{camera_id}")
async def patch_runtime_options(
    camera_id: str,
    body: RuntimeOptionsPatch,
    request: Request,
    mode: str = "site",
):
    changes = {
        key: value
        for key, value in body.model_dump().items()
        if value is not None
    }
    try:
        options = _store(request).update(camera_id, **changes)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    posture_worker = getattr(request.app.state, "posture_worker", None)
    if posture_worker is not None and "posture" in changes:
        posture_worker.set_camera_enabled(camera_id, bool(changes["posture"]))

    worker_id_worker = getattr(request.app.state, "worker_id_worker", None)
    if worker_id_worker is not None and "worker_id" in changes:
        worker_id_worker.set_camera_enabled(
            camera_id,
            bool(changes["worker_id"]),
        )

    return _payload(options, mode)


@router.delete("/{camera_id}")
async def reset_runtime_options(camera_id: str, request: Request, mode: str = "site"):
    options = _store(request).reset(camera_id)
    posture_worker = getattr(request.app.state, "posture_worker", None)
    if posture_worker is not None:
        posture_worker.set_camera_enabled(camera_id, options.posture)
    worker_id_worker = getattr(request.app.state, "worker_id_worker", None)
    if worker_id_worker is not None:
        worker_id_worker.set_camera_enabled(camera_id, options.worker_id)
    return _payload(options, mode)
