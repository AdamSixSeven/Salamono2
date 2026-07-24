"""Persistent directory mapping visible QR identifiers to worker profiles.

The QR decoder deliberately remains independent from personal data: it only
extracts a ``worker_id``.  This store resolves that identifier to the profile
entered by an operator.  SQLite is part of Python's standard library, keeps
the pilot deployment self-contained, and persists inside the existing
``data`` volume used by Docker Compose.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import sqlite3
import threading
import time
from typing import Iterable, Iterator


SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class WorkerRecord:
    worker_id: str
    first_name: str
    last_name: str
    position: str
    department: str
    created_at: float
    updated_at: float

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


class DuplicateWorkerError(ValueError):
    """Raised when an operator tries to create an existing worker ID."""


class WorkerStore:
    """Small thread-safe SQLite repository.

    A short-lived connection is used for every operation.  FastAPI executes
    synchronous route handlers in a worker pool, so this avoids sharing a
    connection between threads while SQLite/WAL handles concurrent readers.
    """

    def __init__(self, path: str):
        self.path = os.fspath(path)
        self._schema_lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Commit/rollback the operation and always release the file handle."""
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _ensure_schema(self) -> None:
        with self._schema_lock, self._connection() as connection:
            # WAL lets frame-processing reads continue while an operator edits
            # the directory from a synchronous API route.
            connection.execute("PRAGMA journal_mode = WAL")
            current_version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if current_version > SCHEMA_VERSION:
                raise RuntimeError(
                    "worker database schema is newer than this application "
                    f"({current_version} > {SCHEMA_VERSION})"
                )
            if current_version < 1:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workers (
                        worker_id TEXT PRIMARY KEY,
                        first_name TEXT NOT NULL,
                        last_name TEXT NOT NULL,
                        position TEXT NOT NULL,
                        department TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_workers_name
                    ON workers(last_name, first_name, worker_id)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_workers_department
                    ON workers(department, last_name, first_name)
                    """
                )
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @staticmethod
    def _record(row: sqlite3.Row | None) -> WorkerRecord | None:
        if row is None:
            return None
        return WorkerRecord(
            worker_id=str(row["worker_id"]),
            first_name=str(row["first_name"]),
            last_name=str(row["last_name"]),
            position=str(row["position"]),
            department=str(row["department"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def create(
        self,
        worker_id: str,
        first_name: str,
        last_name: str,
        position: str,
        department: str,
    ) -> WorkerRecord:
        now = time.time()
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO workers (
                        worker_id, first_name, last_name, position, department,
                        created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        worker_id,
                        first_name,
                        last_name,
                        position,
                        department,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateWorkerError(worker_id) from exc
        worker = self.get(worker_id)
        if worker is None:  # pragma: no cover - defensive guard after INSERT
            raise RuntimeError("worker insert did not persist")
        return worker

    def get(self, worker_id: str) -> WorkerRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT worker_id, first_name, last_name, position, department,
                       created_at, updated_at
                FROM workers
                WHERE worker_id = ?
                """,
                (worker_id,),
            ).fetchone()
        return self._record(row)

    def get_many(self, worker_ids: Iterable[str]) -> dict[str, WorkerRecord]:
        unique_ids = list(dict.fromkeys(str(worker_id) for worker_id in worker_ids))
        if not unique_ids:
            return {}
        placeholders = ",".join("?" for _ in unique_ids)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT worker_id, first_name, last_name, position, department,
                       created_at, updated_at
                FROM workers
                WHERE worker_id IN ({placeholders})
                """,
                unique_ids,
            ).fetchall()
        records = [self._record(row) for row in rows]
        return {
            record.worker_id: record
            for record in records
            if record is not None
        }

    def list(self) -> list[WorkerRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT worker_id, first_name, last_name, position, department,
                       created_at, updated_at
                FROM workers
                ORDER BY last_name COLLATE NOCASE, first_name COLLATE NOCASE,
                         worker_id COLLATE NOCASE
                """
            ).fetchall()
        return [
            record
            for row in rows
            if (record := self._record(row)) is not None
        ]

    def update(
        self,
        worker_id: str,
        first_name: str,
        last_name: str,
        position: str,
        department: str,
    ) -> WorkerRecord | None:
        now = time.time()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE workers
                SET first_name = ?, last_name = ?, position = ?,
                    department = ?, updated_at = ?
                WHERE worker_id = ?
                """,
                (
                    first_name,
                    last_name,
                    position,
                    department,
                    now,
                    worker_id,
                ),
            )
        if cursor.rowcount == 0:
            return None
        return self.get(worker_id)

    def delete(self, worker_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM workers WHERE worker_id = ?",
                (worker_id,),
            )
        return cursor.rowcount > 0
