import json
import os
import threading
from typing import Iterable

from backend.models import AlarmRecord


class AlertStore:
    """Append-only JSONL persistence for alarm records. In-memory cache of the
    most recent `max_memory` records for fast queries + broadcast."""

    def __init__(self, path: str, max_memory: int = 2000):
        self.path = path
        self.max_memory = max_memory
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._records: list[AlarmRecord] = self._load_tail()

    def _load_tail(self) -> list[AlarmRecord]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return []
        out: list[AlarmRecord] = []
        for line in lines[-self.max_memory:]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(AlarmRecord.model_validate_json(line))
            except Exception:
                continue
        return out

    def append(self, record: AlarmRecord) -> None:
        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(record.model_dump_json() + "\n")
            except OSError:
                pass
            self._records.append(record)
            if len(self._records) > self.max_memory:
                self._records = self._records[-self.max_memory:]

    def query(
        self,
        mode: str | None = None,
        severity: str | None = None,
        kind: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AlarmRecord]:
        with self._lock:
            records = list(self._records)
        if mode:
            records = [r for r in records if r.mode == mode]
        if severity:
            records = [r for r in records if r.severity == severity]
        if kind:
            records = [r for r in records if r.kind == kind]
        if since is not None:
            records = [r for r in records if r.timestamp >= since]
        if until is not None:
            records = [r for r in records if r.timestamp <= until]
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[offset:offset + limit]

    def summary(self, since: float | None = None,
                until: float | None = None) -> dict:
        """Aggregate counts for the PIP audit-trail dashboard."""
        with self._lock:
            records = list(self._records)
        if since is not None:
            records = [r for r in records if r.timestamp >= since]
        if until is not None:
            records = [r for r in records if r.timestamp <= until]
        by_sev: dict[str, int] = {}
        by_kind: dict[str, int] = {}
        for r in records:
            by_sev[r.severity] = by_sev.get(r.severity, 0) + 1
            by_kind[r.kind] = by_kind.get(r.kind, 0) + 1
        return {
            "total": len(records),
            "by_severity": by_sev,
            "by_kind": by_kind,
        }

    def get(self, record_id: str) -> AlarmRecord | None:
        with self._lock:
            for r in reversed(self._records):
                if r.id == record_id:
                    return r
        return None

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def extend(self, records: Iterable[AlarmRecord]) -> None:
        for r in records:
            self.append(r)
