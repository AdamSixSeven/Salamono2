from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.models import AlarmRecord

router = APIRouter()


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
    return store.query(
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
    return record


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
    return updated
