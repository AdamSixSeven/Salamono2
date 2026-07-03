from fastapi import APIRouter, Request

from backend.models import AlertOut

router = APIRouter()


@router.get("/alerts", response_model=list[AlertOut])
async def get_alerts(
    request: Request,
    limit: int = 50,
    since: float = 0,
):
    history = request.app.state.alert_history
    filtered = [a for a in history if a.timestamp >= since]
    return filtered[-limit:]


@router.get("/alerts/{alert_id}", response_model=AlertOut | None)
async def get_alert(request: Request, alert_id: str):
    for a in request.app.state.alert_history:
        if a.id == alert_id:
            return a
    return None
