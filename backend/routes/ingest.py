import base64
import time
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile

from backend.danger_rules import DangerEvent
from backend.detector import Detection
from backend.models import AlertOut, AlertSeverity, DetectionOut, FrameResultOut

router = APIRouter()

COLOR_PERSON = (0, 255, 0)
COLOR_VEHICLE = (255, 136, 0)
COLOR_DANGER = (0, 0, 255)
COLOR_WARNING = (0, 165, 255)


def _det_to_out(d: Detection) -> DetectionOut:
    return DetectionOut(
        class_id=d.class_id,
        class_name=d.class_name,
        category=d.category,
        box=list(d.box),
        confidence=round(d.confidence, 3),
    )


def _event_to_alert(e: DangerEvent, alert_id: str,
                     thumb_url: str | None = None) -> AlertOut:
    return AlertOut(
        id=alert_id,
        rule_name=e.rule_name,
        severity=AlertSeverity(e.severity),
        person=_det_to_out(e.person),
        hazard=_det_to_out(e.hazard),
        distance_px=round(e.distance_px, 1),
        overlap_iou=round(e.overlap_iou, 3),
        timestamp=e.frame_timestamp,
        frame_thumbnail_url=thumb_url,
    )


def _annotate(frame: np.ndarray, detections: list[Detection],
              raw_dangers: list[DangerEvent],
              confirmed: list[DangerEvent]) -> np.ndarray:
    out = frame.copy()

    for d in detections:
        x1, y1, x2, y2 = d.box
        color = COLOR_PERSON if d.category == "person" else COLOR_VEHICLE
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    for e in raw_dangers:
        pc = ((e.person.box[0] + e.person.box[2]) // 2,
              (e.person.box[1] + e.person.box[3]) // 2)
        hc = ((e.hazard.box[0] + e.hazard.box[2]) // 2,
              (e.hazard.box[1] + e.hazard.box[3]) // 2)
        color = COLOR_DANGER if e.severity == "DANGER" else COLOR_WARNING
        cv2.line(out, pc, hc, color, 1, cv2.LINE_AA)

    for e in confirmed:
        overlay = out.copy()
        cv2.rectangle(overlay,
                      (e.person.box[0], e.person.box[1]),
                      (e.person.box[2], e.person.box[3]),
                      COLOR_DANGER, -1)
        cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)

        pc = ((e.person.box[0] + e.person.box[2]) // 2,
              (e.person.box[1] + e.person.box[3]) // 2)
        hc = ((e.hazard.box[0] + e.hazard.box[2]) // 2,
              (e.hazard.box[1] + e.hazard.box[3]) // 2)
        cv2.line(out, pc, hc, COLOR_DANGER, 3, cv2.LINE_AA)

    if confirmed:
        cv2.rectangle(out, (0, 0), (out.shape[1], 40), COLOR_DANGER, -1)
        cv2.putText(out, f"ALARM — {len(confirmed)} danger(s)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

    return out


@router.post("/frame", response_model=FrameResultOut)
async def receive_frame(
    request: Request,
    image: UploadFile = File(...),
    camera_id: str = Form(default="cam_default"),
    timestamp: float = Form(default=None),
):
    t0 = time.monotonic()

    raw = await image.read()
    arr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return FrameResultOut(
            frame_id=0, timestamp=0, detections=[], active_dangers=[],
            confirmed_alerts=[], frame_jpeg_b64="", processing_ms=0,
        )

    detections = request.app.state.detector.detect(frame)
    now = timestamp or time.time()

    raw_dangers = request.app.state.danger_detector.evaluate(detections, now)
    confirmed = request.app.state.temporal_filter.update(raw_dangers, now)

    annotated = _annotate(frame, detections, raw_dangers, confirmed)

    alert_outs = []
    for evt in confirmed:
        alert_id = uuid.uuid4().hex[:8]
        thumb_url = request.app.state.frame_store.save(annotated, alert_id)
        alert_out = _event_to_alert(evt, alert_id, thumb_url)
        alert_outs.append(alert_out)
        request.app.state.alert_history.append(alert_out)
        if len(request.app.state.alert_history) > 1000:
            request.app.state.alert_history.pop(0)

    active_outs = []
    for evt in raw_dangers:
        active_outs.append(_event_to_alert(evt, "active"))

    _, jpeg_buf = cv2.imencode(".jpg", annotated,
                               [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64 = base64.b64encode(jpeg_buf).decode()

    processing_ms = (time.monotonic() - t0) * 1000
    request.app.state.frame_counter += 1

    result = FrameResultOut(
        frame_id=request.app.state.frame_counter,
        timestamp=now,
        detections=[_det_to_out(d) for d in detections],
        active_dangers=active_outs,
        confirmed_alerts=alert_outs,
        frame_jpeg_b64=b64,
        processing_ms=round(processing_ms, 1),
    )

    await request.app.state.ws_manager.broadcast_json(result.model_dump())

    return result
