from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json

from fastapi import APIRouter, Request
from fastapi.responses import Response

router = APIRouter()


def _records(request: Request, mode, severity, kind, worker_id, review_status, since, until):
    return request.app.state.alert_store.query(
        mode=mode,
        severity=severity,
        kind=kind,
        worker_id=worker_id,
        review_status=review_status,
        since=since,
        until=until,
        limit=2000,
    )


@router.get("/reports/summary")
def report_summary(
    request: Request,
    since: float | None = None,
    until: float | None = None,
):
    records = _records(request, None, None, None, None, None, since, until)
    hourly: dict[str, int] = {}
    cameras: dict[str, int] = {}
    for record in records:
        hour = datetime.fromtimestamp(record.timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:00Z")
        hourly[hour] = hourly.get(hour, 0) + 1
        cameras[record.camera_id] = cameras.get(record.camera_id, 0) + 1
    summary = request.app.state.alert_store.summary(since=since, until=until)
    summary.update({
        "by_hour_utc": dict(sorted(hourly.items())),
        "by_camera": cameras,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    })
    return summary


@router.get("/reports/export.csv")
def export_csv(
    request: Request,
    mode: str | None = None,
    severity: str | None = None,
    kind: str | None = None,
    worker_id: str | None = None,
    review_status: str | None = None,
    since: float | None = None,
    until: float | None = None,
):
    records = _records(request, mode, severity, kind, worker_id, review_status, since, until)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow([
        "timestamp_iso_utc",
        "timestamp_epoch",
        "mode",
        "kind",
        "severity",
        "rule_name",
        "description",
        "camera_id",
        "worker_id",
        "thumbnail_url",
        "clip_url",
        "review_status",
        "reviewed_at_iso_utc",
        "reviewed_by",
        "review_note",
        "details_json",
    ])
    for record in records:
        severity_value = getattr(record.severity, "value", str(record.severity))
        writer.writerow([
            datetime.fromtimestamp(record.timestamp, tz=timezone.utc).isoformat(),
            record.timestamp,
            record.mode,
            record.kind,
            severity_value,
            record.rule_name,
            record.description,
            record.camera_id,
            record.details.get("worker_id", ""),
            record.thumbnail_url or "",
            record.clip_url or "",
            record.review_status,
            (datetime.fromtimestamp(record.reviewed_at, tz=timezone.utc).isoformat()
             if record.reviewed_at else ""),
            record.reviewed_by or "",
            record.review_note or "",
            json.dumps(record.details, ensure_ascii=False, separators=(",", ":")),
        ])
    filename = f"perimetr-audit-{datetime.now(tz=timezone.utc).date().isoformat()}.csv"
    return Response(
        content=stream.getvalue().encode("utf-8-sig"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/reports/training.jsonl")
def export_training_jsonl(
    request: Request,
    review_status: str = "confirmed",
    since: float | None = None,
    until: float | None = None,
):
    """Export human-reviewed incidents as labels for a future dataset.

    The file contains metadata and evidence URLs, not raw biometric profiles.
    ``false_positive`` can be exported as a useful negative class.
    """
    records = _records(
        request, None, None, None, None, review_status, since, until,
    )
    lines = []
    for record in records:
        lines.append(json.dumps({
            "event_id": record.id,
            "timestamp": record.timestamp,
            "camera_id": record.camera_id,
            "kind": record.kind,
            "rule_name": record.rule_name,
            "severity": getattr(record.severity, "value", str(record.severity)),
            "label": record.review_status,
            "worker_id": record.details.get("worker_id"),
            "thumbnail_url": record.thumbnail_url,
            "clip_url": record.clip_url,
            "details": record.details,
            "review_note": record.review_note,
        }, ensure_ascii=False, separators=(",", ":")))
    filename = f"perimetr-training-{review_status}-{datetime.now(tz=timezone.utc).date().isoformat()}.jsonl"
    return Response(
        content=("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
