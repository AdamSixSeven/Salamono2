"""Thread-safe in-memory cache for worker profiles.

Worker marker scans can return the same identifier for many consecutive
frames.  Resolving every result through SQLite adds avoidable I/O to that hot
path, so this cache sits in front of :class:`backend.worker_store.WorkerStore`.
Both existing profiles and confirmed missing identifiers are cached.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import sqlite3
import threading
import time
from typing import Callable, Iterable

from backend.worker_store import WorkerRecord, WorkerStore


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _CacheEntry:
    profile: WorkerRecord | None
    expires_at: float


@dataclass(frozen=True, slots=True)
class WorkerProfileCacheStats:
    """A cheap, immutable diagnostics snapshot."""

    hits: int
    misses: int
    database_errors: int
    stale_hits: int
    invalidations: int
    entries: int


class WorkerProfileCache:
    """TTL cache backed by ``WorkerStore.get`` and ``WorkerStore.get_many``.

    A single lock deliberately covers a cache miss and its SQLite lookup.  The
    profile table is tiny and this prevents a burst of worker frames from
    issuing duplicate queries for the same identifier.  CRUD routes invalidate
    entries only after their database transaction succeeds.

    If SQLite is temporarily unavailable, a previously cached positive profile
    is returned as stale data.  A failed lookup is retried after a short delay
    rather than being cached for the full profile TTL.
    """

    def __init__(
        self,
        store: WorkerStore,
        ttl_seconds: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        ttl_seconds = float(ttl_seconds)
        if not math.isfinite(ttl_seconds) or ttl_seconds < 0:
            raise ValueError("ttl_seconds must be a finite non-negative number")
        self._store = store
        self._ttl_seconds = ttl_seconds
        self._failure_retry_seconds = min(2.0, max(0.25, ttl_seconds * 0.1))
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: dict[str, _CacheEntry] = {}
        self._hits = 0
        self._misses = 0
        self._database_errors = 0
        self._stale_hits = 0
        self._invalidations = 0

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    @staticmethod
    def _key(worker_id: object) -> str:
        return str(worker_id)

    def get(self, worker_id: str) -> WorkerRecord | None:
        """Return one cached profile, loading it once when absent or expired."""
        key = self._key(worker_id)
        with self._lock:
            now = self._clock()
            previous = self._entries.get(key)
            if previous is not None and now < previous.expires_at:
                self._hits += 1
                return previous.profile

            self._misses += 1
            try:
                profile = self._store.get(key)
            except sqlite3.Error:
                return self._after_database_error(key, previous, now)

            self._entries[key] = _CacheEntry(
                profile=profile,
                expires_at=now + self._ttl_seconds,
            )
            return profile

    def get_many(self, worker_ids: Iterable[str]) -> dict[str, WorkerRecord]:
        """Resolve unique identifiers with one batched SQLite query on misses."""
        keys = list(dict.fromkeys(self._key(worker_id) for worker_id in worker_ids))
        if not keys:
            return {}

        with self._lock:
            now = self._clock()
            result: dict[str, WorkerRecord] = {}
            missed: list[str] = []
            previous_by_key: dict[str, _CacheEntry | None] = {}

            for key in keys:
                previous = self._entries.get(key)
                if previous is not None and now < previous.expires_at:
                    self._hits += 1
                    if previous.profile is not None:
                        result[key] = previous.profile
                    continue
                self._misses += 1
                missed.append(key)
                previous_by_key[key] = previous

            if not missed:
                return result

            try:
                loaded = self._store.get_many(missed)
            except sqlite3.Error:
                self._record_database_error(missed)
                for key in missed:
                    previous = previous_by_key[key]
                    if previous is not None and previous.profile is not None:
                        self._stale_hits += 1
                        result[key] = previous.profile
                    self._entries[key] = _CacheEntry(
                        profile=previous.profile if previous is not None else None,
                        expires_at=now + self._failure_retry_seconds,
                    )
                return result

            for key in missed:
                profile = loaded.get(key)
                self._entries[key] = _CacheEntry(
                    profile=profile,
                    expires_at=now + self._ttl_seconds,
                )
                if profile is not None:
                    result[key] = profile
            return result

    def invalidate(self, worker_id: str) -> None:
        """Remove one profile after a successful create, update or delete."""
        key = self._key(worker_id)
        with self._lock:
            self._entries.pop(key, None)
            self._invalidations += 1

    def clear(self) -> None:
        """Invalidate every positive and negative cached profile."""
        with self._lock:
            self._entries.clear()
            self._invalidations += 1

    def stats(self) -> WorkerProfileCacheStats:
        with self._lock:
            return WorkerProfileCacheStats(
                hits=self._hits,
                misses=self._misses,
                database_errors=self._database_errors,
                stale_hits=self._stale_hits,
                invalidations=self._invalidations,
                entries=len(self._entries),
            )

    def _after_database_error(
        self,
        key: str,
        previous: _CacheEntry | None,
        now: float,
    ) -> WorkerRecord | None:
        self._record_database_error([key])
        profile = previous.profile if previous is not None else None
        if profile is not None:
            self._stale_hits += 1
        self._entries[key] = _CacheEntry(
            profile=profile,
            expires_at=now + self._failure_retry_seconds,
        )
        return profile

    def _record_database_error(self, worker_ids: list[str]) -> None:
        self._database_errors += 1
        logger.warning(
            "Worker profile lookup failed for %d identifier(s); "
            "using cached data when available",
            len(worker_ids),
            exc_info=True,
        )
