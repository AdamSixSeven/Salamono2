from __future__ import annotations

from collections import Counter
import re

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import Response

from backend.models import WorkerCreateIn, WorkerProfileOut, WorkerUpdateIn
from backend.worker_store import (
    DuplicateWorkerError,
    WorkerRecord,
    WorkerStore,
)
from config import CONFIG
from backend.worker_tags import render_worker_marker, worker_marker_id

router = APIRouter()


def _profile_out(worker: WorkerRecord) -> WorkerProfileOut:
    return WorkerProfileOut(
        worker_id=worker.worker_id,
        first_name=worker.first_name,
        last_name=worker.last_name,
        full_name=worker.full_name,
        position=worker.position,
        department=worker.department,
        created_at=worker.created_at,
        updated_at=worker.updated_at,
    )


def _worker_store(request: Request) -> WorkerStore:
    """Return the lifespan-managed store, with a test/embedded-app fallback."""
    store = getattr(request.app.state, "worker_store", None)
    if store is None:
        store = WorkerStore(CONFIG.worker_id.database_path)
        request.app.state.worker_store = store
    return store


def _invalidate_worker_profile_cache(request: Request, worker_id: str) -> None:
    """Keep profile and marker lookup caches coherent after successful CRUD."""
    cache = getattr(request.app.state, "worker_profile_cache", None)
    if cache is not None:
        cache.invalidate(worker_id)
    identifier = getattr(request.app.state, "worker_identifier", None)
    invalidate_marker_index = getattr(
        identifier,
        "invalidate_marker_index",
        None,
    )
    if callable(invalidate_marker_index):
        invalidate_marker_index()


def _clean_path_worker_id(worker_id: str) -> str:
    worker_id = worker_id.strip()
    if (
        not worker_id
        or len(worker_id) > 64
        or any(ord(char) < 32 for char in worker_id)
    ):
        raise HTTPException(400, "invalid worker_id")
    return worker_id


def _validated_worker_id(worker_id: str) -> str:
    worker_id = worker_id.strip()
    if (
        not worker_id
        or len(worker_id) > 64
        or any(ord(char) < 32 for char in worker_id)
    ):
        raise HTTPException(400, "invalid worker_id")
    return worker_id


def _safe_worker_filename_id(worker_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", worker_id).strip("._") or "worker"



def _centered_text(
    canvas: np.ndarray,
    text: str,
    y: int,
    *,
    font_scale: float,
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, _text_height), _baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )
    x = max(20, (canvas.shape[1] - text_width) // 2)
    cv2.putText(
        canvas,
        text,
        (x, y),
        font,
        font_scale,
        0,
        thickness,
        cv2.LINE_AA,
    )


@router.get("/worker-tag.png")
def worker_tag_png(
    worker_id: str = Query(..., min_length=1, max_length=64),
    download: bool = Query(False),
) -> Response:
    """Return the compact high-contrast worker marker as printable PNG."""
    worker_id = _validated_worker_id(worker_id)
    safe_filename_id = _safe_worker_filename_id(worker_id)
    marker_id = worker_marker_id(worker_id)
    marker = render_worker_marker(marker_id, side_pixels=1000, border_bits=1)

    # Add a plain white quiet zone around the marker; this significantly helps
    # detection under blur and at long range.
    quiet = 120
    marker = cv2.copyMakeBorder(marker, quiet, quiet, quiet, quiet, cv2.BORDER_CONSTANT, value=255)

    margin = 120
    header_height = 130
    caption_height = 270
    canvas_width = marker.shape[1] + margin * 2
    canvas_height = header_height + marker.shape[0] + caption_height + margin
    canvas = np.full((canvas_height, canvas_width), 255, dtype=np.uint8)

    marker_y = header_height
    marker_x = (canvas_width - marker.shape[1]) // 2
    canvas[marker_y:marker_y + marker.shape[0], marker_x:marker_x + marker.shape[1]] = marker

    printable_id = "".join(char if 32 <= ord(char) <= 126 else "_" for char in worker_id)
    _centered_text(canvas, "PERIMETR - WORKER TAG", 78, font_scale=1.25, thickness=3)
    _centered_text(canvas, f"WORKER ID: {printable_id}", marker_y + marker.shape[0] + 95, font_scale=1.7, thickness=4)
    _centered_text(canvas, f"MARKER ID: {marker_id}", marker_y + marker.shape[0] + 160, font_scale=1.2, thickness=3)
    _centered_text(canvas, "PLACE ON VEST OR BACK", marker_y + marker.shape[0] + 225, font_scale=0.95, thickness=2)

    encoded, buffer = cv2.imencode('.png', canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not encoded:  # pragma: no cover
        raise HTTPException(500, "Could not generate worker marker PNG")

    disposition = "attachment" if download else "inline"
    filename = f"worker-{safe_filename_id}.png"
    return Response(
        content=buffer.tobytes(),
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'{disposition}; filename="{filename}"',
            "X-Worker-Marker-Id": str(marker_id),
            "X-Worker-Tag-Type": "aruco-4x4",
        },
    )


@router.get("/workers/summary")
def worker_summary(
    request: Request,
    since: float | None = None,
    until: float | None = None,
):
    records = request.app.state.alert_store.query(
        since=since,
        until=until,
        limit=2000,
    )
    by_worker: Counter[str] = Counter()
    danger_by_worker: Counter[str] = Counter()
    for record in records:
        worker_id = record.details.get("worker_id")
        if not worker_id:
            continue
        worker_id = str(worker_id)
        by_worker[worker_id] += 1
        if str(record.severity) == "DANGER" or getattr(record.severity, "value", None) == "DANGER":
            danger_by_worker[worker_id] += 1
    store = getattr(request.app.state, "worker_store", None)
    profiles = store.get_many(by_worker.keys()) if store is not None else {}
    rows = []
    for worker_id, count in by_worker.most_common():
        row = {
            "worker_id": worker_id,
            "events": count,
            "danger_events": danger_by_worker.get(worker_id, 0),
        }
        profile = profiles.get(worker_id)
        if profile is not None:
            row.update({
                "registered": True,
                "first_name": profile.first_name,
                "last_name": profile.last_name,
                "full_name": profile.full_name,
                "position": profile.position,
                "department": profile.department,
            })
        rows.append(row)
    return {
        "workers": rows
    }


@router.get("/workers", response_model=list[WorkerProfileOut])
def list_workers(request: Request):
    return [_profile_out(worker) for worker in _worker_store(request).list()]


@router.post(
    "/workers",
    response_model=WorkerProfileOut,
    status_code=status.HTTP_201_CREATED,
)
def create_worker(request: Request, payload: WorkerCreateIn):
    try:
        worker = _worker_store(request).create(
            worker_id=payload.worker_id,
            first_name=payload.first_name,
            last_name=payload.last_name,
            position=payload.position,
            department=payload.department,
        )
    except DuplicateWorkerError:
        raise HTTPException(409, "worker_id already exists")
    _invalidate_worker_profile_cache(request, worker.worker_id)
    return _profile_out(worker)


# Keep dynamic routes below /workers/summary so "summary" cannot be consumed
# as a worker identifier by Starlette's first-match routing.
@router.get("/workers/{worker_id}", response_model=WorkerProfileOut)
def get_worker(request: Request, worker_id: str):
    worker_id = _clean_path_worker_id(worker_id)
    worker = _worker_store(request).get(worker_id)
    if worker is None:
        raise HTTPException(404, "worker not found")
    return _profile_out(worker)


@router.put("/workers/{worker_id}", response_model=WorkerProfileOut)
def update_worker(request: Request, worker_id: str, payload: WorkerUpdateIn):
    worker_id = _clean_path_worker_id(worker_id)
    worker = _worker_store(request).update(
        worker_id=worker_id,
        first_name=payload.first_name,
        last_name=payload.last_name,
        position=payload.position,
        department=payload.department,
    )
    if worker is None:
        raise HTTPException(404, "worker not found")
    _invalidate_worker_profile_cache(request, worker_id)
    return _profile_out(worker)


@router.delete("/workers/{worker_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_worker(request: Request, worker_id: str):
    worker_id = _clean_path_worker_id(worker_id)
    if not _worker_store(request).delete(worker_id):
        raise HTTPException(404, "worker not found")
    _invalidate_worker_profile_cache(request, worker_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
