"""Append-only SQLite persistence for Central AI agent outputs (slice A: explain).

Reuses the SAME database file as the assessment store (``<data_dir>/state/telltale.db``)
and its versioned migration bookkeeping (the shared ``schema_migrations`` table), with
one addition: version 3 creates the NEW ``ai_outputs`` table only. Existing tables
(``alerts``, ``assessments``) and their records are never touched, and assessment
snapshots are never mutated by this store.

IMMUTABILITY SCOPE (documented on purpose): insert/get/close only - no replace,
update or delete methods - and duplicate server-generated ids are rejected with the
typed conflict :class:`AiOutputAlreadyExists`. That is application-level immutability
for well-behaved clients; it is NOT tamper-proof storage against a database
administrator with direct file access.

Nothing here stores raw prompts, secrets, raw audio or credentials: only the prompt
DIGEST (sha256) and the validated structured output are kept.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..schemas.ai import AiOutputRecord


class AiOutputAlreadyExists(Exception):
    """Typed conflict: inserting an ai_output_id that is already stored."""

    def __init__(self, ai_output_id: Any) -> None:
        self.ai_output_id = ai_output_id
        super().__init__(f"ai output already exists: {ai_output_id}")


def _migration_ai_outputs_v3(conn: sqlite3.Connection) -> None:
    # New table only; assessment/alerts tables and rows are never altered. The full
    # record is kept as JSON (same pattern as the assessment store) plus the audited
    # columns routes look up by.
    conn.execute(
        "CREATE TABLE ai_outputs ("
        "ai_output_id TEXT PRIMARY KEY, "
        "owner TEXT NOT NULL DEFAULT '', "
        "kind TEXT NOT NULL, "
        "assessment_ids TEXT NOT NULL, "
        "provider TEXT NOT NULL, "
        "model TEXT, "
        "prompt_version TEXT NOT NULL, "
        "prompt_sha256 TEXT NOT NULL, "
        "response_text TEXT NOT NULL, "
        "validation_status TEXT NOT NULL, "
        "created_at TEXT NOT NULL, "
        "json TEXT NOT NULL)"
    )


# (version, name, runner) - version 3 follows the assessment store's 1 and 2 in the
# shared schema_migrations recorder table. Append new migrations, never edit applied ones.
MIGRATIONS: tuple[tuple[int, str, Callable[[sqlite3.Connection], None]], ...] = (
    (3, "ai_outputs_v3", _migration_ai_outputs_v3),
)


def run_migrations(conn: sqlite3.Connection) -> list[int]:
    """Apply pending migrations, each in its own transaction. Returns new versions."""
    # The recorder table itself is idempotent bookkeeping (IF NOT EXISTS is correct
    # here); versioned DDL below is gated by recorded versions, not by IF NOT EXISTS.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version INTEGER PRIMARY KEY, "
        "name TEXT NOT NULL, "
        "applied_at TEXT NOT NULL)"
    )
    recorded = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    newly_applied: list[int] = []
    for version, name, runner in MIGRATIONS:
        if version in recorded:
            continue
        conn.execute("BEGIN IMMEDIATE")
        try:
            runner(conn)
            conn.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                (version, name, datetime.now(timezone.utc).isoformat()),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        newly_applied.append(version)
    return newly_applied


class AiOutputStore:
    """Append-only store for validated AI outputs: insert / get / close only."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None => autocommit; transactions are explicit below.
        self._db = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        with self._lock:
            run_migrations(self._db)

    def insert(self, record: AiOutputRecord, *, owner: str = "") -> None:
        """Store one AI output record once. ``owner`` is written at insert time only
        (never updated). Duplicate ai_output_id -> typed conflict, nothing persisted."""
        # Serialize a deep copy so later mutation of the caller's object can never
        # change what was written.
        payload = record.model_copy(deep=True)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute(
                    "INSERT INTO ai_outputs "
                    "(ai_output_id, owner, kind, assessment_ids, provider, model, prompt_version, "
                    "prompt_sha256, response_text, validation_status, created_at, json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(payload.ai_output_id),
                        owner,
                        payload.kind,
                        json.dumps(payload.assessment_ids),
                        payload.provider,
                        payload.model,
                        payload.prompt_version,
                        payload.prompt_sha256,
                        payload.response_text,
                        payload.validation_status,
                        # Fixed microsecond precision: ISO text ordering must never
                        # depend on mixed precision.
                        payload.created_at.isoformat(timespec="microseconds"),
                        payload.model_dump_json(),
                    ),
                )
                self._db.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                self._db.execute("ROLLBACK")
                raise AiOutputAlreadyExists(payload.ai_output_id) from exc
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def get(self, ai_output_id: Any, *, owner: str | None = None) -> AiOutputRecord | None:
        """Fresh parse per call, so callers cannot mutate stored content.

        ``owner`` scopes the lookup (authenticated routes MUST pass it); a mismatching
        owner returns None, so cross-owner reads behave as 404.
        """
        query = "SELECT json FROM ai_outputs WHERE ai_output_id = ?"
        params: list[Any] = [str(ai_output_id)]
        if owner is not None:
            query += " AND owner = ?"
            params.append(owner)
        with self._lock:
            row = self._db.execute(query, params).fetchone()
        if row is None:
            return None
        return AiOutputRecord.model_validate_json(row[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()


__all__ = [
    "AiOutputAlreadyExists",
    "AiOutputStore",
    "MIGRATIONS",
    "run_migrations",
]


# Guard against accidental API growth: append-only means insert/get/close only.
for _name in ("update", "replace", "delete", "upsert", "remove", "set"):
    if hasattr(AiOutputStore, _name):  # pragma: no cover - fails loudly at import
        raise RuntimeError(f"AiOutputStore must stay append-only (found '{_name}')")
del _name
