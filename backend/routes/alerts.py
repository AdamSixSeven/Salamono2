import os
from urllib.parse import quote, unquote, urlparse

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.models import AlarmRecord
from config import CONFIG

router = APIRouter()


def _history_record(request: Request, record: AlarmRecord) -> AlarmRecord:
    """Expose browser URLs and current worker profiles, never server paths."""
    filename = None
    if record.clip_url:
        filename = os.path.basename(unquote(urlparse(record.clip_url).path))
    recorder = getattr(request.app.state, "evidence_recorder", None)
    clips_dir = getattr(recorder, "base_dir", CONFIG.evidence.clips_dir)
    browser_filename = (
        f"{os.path.splitext(filename)[0]}.browser.mp4" if filename else None
    )
    if browser_filename and os.path.isfile(os.path.join(clips_dir, browser_filename)):
        filename = browser_filename
    available = bool(
        filename
        and os.path.isfile(os.path.join(clips_dir, filename))
        and os.path.getsize(os.path.join(clips_dir, filename)) > 0
    )
    clip_url = f"/static/clips/{quote(filename)}" if available else None
    worker = None
    worker_id = record.details.get("worker_id")
    if worker_id:
        profile_cache = getattr(request.app.state, "worker_profile_cache", None)
        profile = profile_cache.get(str(worker_id)) if profile_cache is not None else None
        if profile is not None:
            worker = {
                "worker_id": profile.worker_id,
                "first_name": profile.first_name,
                "last_name": profile.last_name,
            }
    return record.model_copy(update={
        "clip_available": available,
        "clip_url": clip_url,
        "clip_filename": filename if available else None,
        "snapshot_url": record.thumbnail_url,
        "worker": worker,
    })


class ReviewUpdate(BaseModel):
    status: str
    reviewed_by: str | None = Field(default=None, max_length=80)
    note: str | None = Field(default=None, max_length=1000)


@router.get("/alerts", response_model=list[AlarmRecord])
async def get_alerts(
    request: Request,
    mode: str | None = None,
    severity: str | None = None,
    kind: str | None = None,
    worker_id: str | None = None,
    review_status: str | None = None,
    since: float | None = None,
    until: float | None = None,
    limit: int = 100,
    offset: int = 0,
):
    store = request.app.state.alert_store
    records = store.query(
        mode=mode,
        severity=severity,
        kind=kind,
        worker_id=worker_id,
        review_status=review_status,
        since=since,
        until=until,
        limit=max(1, min(limit, 500)),
        offset=max(0, offset),
    )
    return [_history_record(request, record) for record in records]


@router.get("/alerts/summary")
async def get_alerts_summary(
    request: Request,
    mode: str | None = None,
    severity: str | None = None,
    kind: str | None = None,
    worker_id: str | None = None,
    review_status: str | None = None,
    since: float | None = None,
    until: float | None = None,
):
    return request.app.state.alert_store.summary(
        mode=mode,
        severity=severity,
        kind=kind,
        worker_id=worker_id,
        review_status=review_status,
        since=since,
        until=until,
    )


@router.get("/alerts/{record_id}", response_model=AlarmRecord)
async def get_alert(request: Request, record_id: str):
    record = request.app.state.alert_store.get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="not found")
    return _history_record(request, record)


@router.patch("/alerts/{record_id}/review", response_model=AlarmRecord)
async def review_alert(request: Request, record_id: str, payload: ReviewUpdate):
    try:
        updated = request.app.state.alert_store.update_review(
            record_id,
            status=payload.status,
            reviewed_by=payload.reviewed_by,
            note=payload.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if updated is None:
        raise HTTPException(status_code=404, detail="not found")
    return _history_record(request, updated)
