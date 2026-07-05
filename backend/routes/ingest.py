import base64
import time
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile

from backend.danger_rules import DangerEvent
from backend.detector import Detection
from backend.models import (
    AlarmRecord,
    AlertOut,
    AlertSeverity,
    DetectionOut,
    FrameResultOut,
    PPECheckOut,
)
from backend.ppe_rules import PPEEvent

SITE_RULE_DESCRIPTIONS = {
    "person_vehicle_overlap": "Osoba w strefie pojazdu",
    "person_near_vehicle": "Osoba blisko pojazdu",
}


def _site_record(alert: AlertOut, camera_id: str) -> AlarmRecord:
    return AlarmRecord(
        id=alert.id,
        timestamp=alert.timestamp,
        mode="site",
        kind="site_hazard",
        severity=alert.severity,
        rule_name=alert.rule_name,
        description=SITE_RULE_DESCRIPTIONS.get(alert.rule_name, alert.rule_name),
        camera_id=camera_id,
        thumbnail_url=alert.frame_thumbnail_url,
        details={
            "distance_px": alert.distance_px,
            "overlap_iou": alert.overlap_iou,
            "person_confidence": alert.person.confidence,
            "hazard_confidence": alert.hazard.confidence,
            "person_box": alert.person.box,
            "hazard_box": alert.hazard.box,
            "hazard_class": alert.hazard.class_name,
        },
    )


def _ppe_record(check: PPECheckOut, camera_id: str) -> AlarmRecord:
    missing_pretty = {"hardhat": "kask", "vest": "kamizelka"}
    missing_labels = [missing_pretty.get(m, m) for m in check.missing]
    if missing_labels:
        desc = "Brak PPE: " + " + ".join(missing_labels)
    else:
        desc = "PPE OK"
    return AlarmRecord(
        id=check.id,
        timestamp=check.timestamp,
        mode="checkpoint",
        kind="ppe_missing",
        severity=check.severity,
        rule_name="missing_" + "_".join(check.missing) if check.missing else "ppe_ok",
        description=desc,
        camera_id=camera_id,
        thumbnail_url=check.frame_thumbnail_url,
        details={
            "missing": check.missing,
            "has_hardhat": check.has_hardhat,
            "has_vest": check.has_vest,
            "person_confidence": check.person.confidence,
            "person_box": check.person.box,
        },
    )

router = APIRouter()

COLOR_PERSON = (0, 255, 0)
COLOR_VEHICLE = (255, 136, 0)
COLOR_DANGER = (0, 0, 255)
COLOR_WARNING = (0, 165, 255)
COLOR_HARDHAT = (255, 255, 0)
COLOR_VEST = (0, 255, 255)
COLOR_OK = (0, 200, 0)


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


def _ppe_to_out(e: PPEEvent, check_id: str,
                thumb_url: str | None = None) -> PPECheckOut:
    return PPECheckOut(
        id=check_id,
        severity=AlertSeverity(e.severity),
        missing=e.missing,
        has_hardhat=e.hardhat is not None,
        has_vest=e.vest is not None,
        person=_det_to_out(e.person),
        timestamp=e.frame_timestamp,
        frame_thumbnail_url=thumb_url,
    )


def _annotate_site(frame: np.ndarray, detections: list[Detection],
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

    if confirmed:
        cv2.rectangle(out, (0, 0), (out.shape[1], 40), COLOR_DANGER, -1)
        cv2.putText(out, f"ALARM — {len(confirmed)} danger(s)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

    return out


def _annotate_ppe(frame: np.ndarray, detections: list[Detection],
                  events: list[PPEEvent]) -> np.ndarray:
    out = frame.copy()

    for d in detections:
        x1, y1, x2, y2 = d.box
        if d.category == "person":
            color = COLOR_PERSON
        elif d.category == "hardhat":
            color = COLOR_HARDHAT
        elif d.category == "vest":
            color = COLOR_VEST
        else:
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    banner_color = None
    banner_text = None
    for e in events:
        if not e.confirmed:
            continue
        color = COLOR_DANGER if e.missing else COLOR_OK
        overlay = out.copy()
        cv2.rectangle(overlay,
                      (e.person.box[0], e.person.box[1]),
                      (e.person.box[2], e.person.box[3]),
                      color, -1)
        cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
        if e.missing:
            banner_color = COLOR_DANGER
            banner_text = "MISSING: " + " + ".join(m.upper() for m in e.missing)
        else:
            banner_color = COLOR_OK
            banner_text = "PPE OK"

    if banner_color and banner_text:
        cv2.rectangle(out, (0, 0), (out.shape[1], 40), banner_color, -1)
        cv2.putText(out, banner_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return out


async def _handle_site(request: Request, frame: np.ndarray, now: float,
                        t0: float, camera_id: str) -> FrameResultOut:
    detector = request.app.state.detector
    danger_detector = request.app.state.danger_detector
    temporal_filter = request.app.state.temporal_filter
    frame_store = request.app.state.frame_store
    alert_store = request.app.state.alert_store

    detections = detector.detect(frame)
    raw_dangers = danger_detector.evaluate(detections, now)
    confirmed = temporal_filter.update(raw_dangers, now)
    annotated = _annotate_site(frame, detections, raw_dangers, confirmed)

    alert_outs = []
    for evt in confirmed:
        alert_id = uuid.uuid4().hex[:8]
        thumb_url = frame_store.save(annotated, alert_id)
        alert = _event_to_alert(evt, alert_id, thumb_url)
        alert_outs.append(alert)
        alert_store.append(_site_record(alert, camera_id))

    active_outs = [_event_to_alert(e, "active") for e in raw_dangers]

    _, jpeg_buf = cv2.imencode(".jpg", annotated,
                               [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64 = base64.b64encode(jpeg_buf).decode()

    processing_ms = (time.monotonic() - t0) * 1000
    request.app.state.frame_counter += 1

    return FrameResultOut(
        frame_id=request.app.state.frame_counter,
        timestamp=now,
        mode="site",
        detections=[_det_to_out(d) for d in detections],
        active_dangers=active_outs,
        confirmed_alerts=alert_outs,
        frame_jpeg_b64=b64,
        processing_ms=round(processing_ms, 1),
    )


async def _handle_checkpoint(request: Request, frame: np.ndarray, now: float,
                              t0: float, camera_id: str) -> FrameResultOut:
    detector = request.app.state.ppe_detector
    checker = request.app.state.ppe_checker
    frame_store = request.app.state.frame_store
    alert_store = request.app.state.alert_store

    detections = detector.detect(frame)
    events = checker.evaluate(detections, frame_h=frame.shape[0],
                              frame_timestamp=now)
    confirmed = checker.confirm(events, now)
    annotated = _annotate_ppe(frame, detections, confirmed)

    check_outs = []
    for evt in confirmed:
        cid = uuid.uuid4().hex[:8]
        thumb_url = frame_store.save(annotated, cid)
        out = _ppe_to_out(evt, cid, thumb_url)
        check_outs.append(out)
        if evt.missing:
            alert_store.append(_ppe_record(out, camera_id))

    _, jpeg_buf = cv2.imencode(".jpg", annotated,
                               [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64 = base64.b64encode(jpeg_buf).decode()

    processing_ms = (time.monotonic() - t0) * 1000
    request.app.state.frame_counter += 1

    return FrameResultOut(
        frame_id=request.app.state.frame_counter,
        timestamp=now,
        mode="checkpoint",
        detections=[_det_to_out(d) for d in detections],
        ppe_checks=check_outs,
        frame_jpeg_b64=b64,
        processing_ms=round(processing_ms, 1),
    )


@router.post("/frame", response_model=FrameResultOut)
async def receive_frame(
    request: Request,
    image: UploadFile = File(...),
    camera_id: str = Form(default="cam_default"),
    timestamp: float = Form(default=None),
    mode: str = Form(default="site"),
):
    t0 = time.monotonic()

    raw = await image.read()
    arr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return FrameResultOut(
            frame_id=0, timestamp=0, mode=mode,
            detections=[], frame_jpeg_b64="", processing_ms=0,
        )

    now = timestamp or time.time()

    if mode == "checkpoint" and request.app.state.ppe_detector is not None:
        result = await _handle_checkpoint(request, frame, now, t0, camera_id)
    else:
        result = await _handle_site(request, frame, now, t0, camera_id)

    await request.app.state.ws_manager.broadcast_json(result.model_dump())
    return result
