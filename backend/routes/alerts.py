from fastapi import APIRouter, HTTPException, Request

from backend.models import AlarmRecord

router = APIRouter()


@router.get("/alerts", response_model=list[AlarmRecord])
async def get_alerts(
    request: Request,
    mode: str | None = None,
    severity: str | None = None,
    kind: str | None = None,
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
        since=since,
        until=until,
        limit=max(1, min(limit, 500)),
        offset=max(0, offset),
    )


@router.get("/alerts/summary")
async def get_alerts_summary(
    request: Request,
    since: float | None = None,
    until: float | None = None,
):
    return request.app.state.alert_store.summary(since=since, until=until)


@router.get("/alerts/{record_id}", response_model=AlarmRecord)
async def get_alert(request: Request, record_id: str):
    record = request.app.state.alert_store.get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="not found")
    return record
