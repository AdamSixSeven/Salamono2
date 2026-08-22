"""Debug endpoints — inject a fake person detection into the next N frames.

Adam wanted to test the alarm flow without physically standing in the
zone. This lets you POST a bounding box (normalized 0-1) and the backend
pretends YOLO detected a person there for the next N frames. Zone
breach, temporal filter, alerts, voice and history all fire naturally.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


class InjectPersonPayload(BaseModel):
    # bbox in normalized coords 0..1 — [x1, y1, x2, y2]
    box_norm: list[float] = Field(..., min_length=4, max_length=4)
    # how many upcoming frames to inject into
    frames: int = Field(default=10, ge=1, le=1000)
    # detection confidence to report
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


def _validate_box(box: list[float]) -> None:
    if len(box) != 4:
        raise HTTPException(400, "box_norm must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = box
    if not all(0.0 <= v <= 1.0 for v in box):
        raise HTTPException(400, "box_norm coords must be in [0, 1]")
    if x1 >= x2 or y1 >= y2:
        raise HTTPException(400, "box_norm must have x1<x2 and y1<y2")


@router.post("/debug/inject-person")
async def inject_person(request: Request, payload: InjectPersonPayload):
    _validate_box(payload.box_norm)
    request.app.state.debug_inject_person = {
        "box_norm": list(payload.box_norm),
        "remaining": payload.frames,
        "confidence": payload.confidence,
    }
    return {
        "status": "armed",
        "box_norm": payload.box_norm,
        "frames": payload.frames,
    }


@router.get("/debug/inject-person")
async def inject_person_status(request: Request):
    state = getattr(request.app.state, "debug_inject_person", None)
    if not state:
        return {"status": "idle"}
    return {
        "status": "armed",
        "remaining": state["remaining"],
        "box_norm": state["box_norm"],
    }


@router.delete("/debug/inject-person")
async def inject_person_clear(request: Request):
    request.app.state.debug_inject_person = None
    return {"status": "cleared"}
