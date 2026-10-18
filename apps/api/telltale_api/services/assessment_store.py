"""Append-only SQLite persistence for immutable assessments.

Reuses the existing SQLite database architecture (same ``<data_dir>/state/telltale.db``
layout as :class:`telltale_api.services.alerts.SqliteAlertStore`), but with explicit,
transaction-based, versioned migrations:

- a ``schema_migrations`` recorder table stores ``(version, name, applied_at)``;
- each pending migration runs its DDL and its version row inside ONE transaction
  (BEGIN IMMEDIATE ... COMMIT, ROLLBACK on error);
- migrations are scoped to NEW tables only - the existing ``alerts`` tables/records
  are never touched;
- versioned DDL deliberately does NOT rely on ``CREATE TABLE IF NOT EXISTS``: the
  version row is the source of truth, and re-running is a recorded no-op.

IMMUTABILITY SCOPE (documented on purpose): this store exposes insert/get/list only -
no replace, update or delete methods - and duplicate server-generated ids are rejected
with the typed conflict :class:`AssessmentAlreadyExists`. That is APPLICATION-LEVEL
immutability for well-behaved clients; it is NOT tamper-proof storage against a
database administrator with direct file access.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..schemas.assessment import Assessment

DEFAULT_PAGE = 50
MAX_PAGE = 200


class AssessmentAlreadyExists(Exception):
    """Typed conflict: inserting an assessment_id that is already stored."""

    def __init__(self, assessment_id: Any) -> None:
        self.assessment_id = assessment_id
        super().__init__(f"assessment already exists: {assessment_id}")


def _migration_v1(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE assessments ("
        "assessment_id TEXT PRIMARY KEY, "
        "machine_id TEXT NOT NULL, "
        "source_mode TEXT NOT NULL, "
        "schema_version TEXT NOT NULL, "
        "rule_version TEXT NOT NULL, "
        "created_at TEXT NOT NULL, "
        "json TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX idx_assessments_machine_created_at "
        "ON assessments (machine_id, created_at)"
    )


def _migration_v2_owner(conn: sqlite3.Connection) -> None:
    # Owner is ACCESS metadata (who the record belongs to), written once at insert
    # and never updated. Existing rows keep owner='' and stay inaccessible through
    # the authenticated assessment routes; they are never reassigned.
    conn.execute("ALTER TABLE assessments ADD COLUMN owner TEXT NOT NULL DEFAULT ''")


# (version, name, runner) - append new migrations, never edit applied ones.
MIGRATIONS: tuple[tuple[int, str, Callable[[sqlite3.Connection], None]], ...] = (
    (1, "assessments_v1", _migration_v1),
    (2, "assessments_owner_v2", _migration_v2_owner),
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


class AssessmentStore:
    """Append-only store: insert / get / list (bounded pagination, stable order)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None => autocommit; transactions are explicit below.
        self._db = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        with self._lock:
            run_migrations(self._db)

    def insert(self, assessment: Assessment, *, owner: str = "") -> None:
        """Store an immutable snapshot once. ``owner`` is written at insert time
        only (never updated). Duplicate assessment_id -> typed conflict."""
        # Serialize a deep copy so later mutation of the caller's object can never
        # change what was written.
        payload = assessment.model_copy(deep=True).model_dump_json()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute(
                    "INSERT INTO assessments "
                    "(assessment_id, machine_id, source_mode, schema_version, "
                    "rule_version, created_at, json, owner) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(assessment.assessment_id),
                        assessment.machine_id,
                        assessment.source_mode,
                        assessment.schema_version,
                        assessment.rule_version,
                        # Fixed microsecond precision: ISO text ordering must never
                        # depend on mixed precision ("14:00:00+00:00" vs
                        # "14:00:00.500000+00:00" is fragile; "+00:00" < "." only
                        # happens to sort right). All rows share one exact format.
                        assessment.created_at.isoformat(timespec="microseconds"),
                        payload,
                        owner,
                    ),
                )
                self._db.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                self._db.execute("ROLLBACK")
                raise AssessmentAlreadyExists(assessment.assessment_id) from exc
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def get(self, assessment_id: Any, *, owner: str | None = None) -> Assessment | None:
        """Fresh parse per call, so callers cannot mutate stored content.

        ``owner`` scopes the lookup (authenticated routes MUST pass it); a
        mismatching owner returns None, so cross-owner reads behave as 404.
        ``owner=None`` is for unauthenticated internal/test use only.
        """
        query = "SELECT json FROM assessments WHERE assessment_id = ?"
        params: list[Any] = [str(assessment_id)]
        if owner is not None:
            query += " AND owner = ?"
            params.append(owner)
        with self._lock:
            row = self._db.execute(query, params).fetchone()
        if row is None:
            return None
        return Assessment.model_validate_json(row[0])

    def list(
        self,
        *,
        owner: str | None = None,
        machine_id: str | None = None,
        limit: int = DEFAULT_PAGE,
        offset: int = 0,
    ) -> list[Assessment]:
        """Bounded pagination with a stable total order (created_at DESC, id DESC).

        ``owner`` MUST be passed by authenticated routes: the list is always
        owner-filtered, never a shared view.
        """
        limit = max(1, min(int(limit), MAX_PAGE))
        offset = max(0, int(offset))
        clauses: list[str] = []
        params: list[Any] = []
        if owner is not None:
            clauses.append("owner = ?")
            params.append(owner)
        if machine_id is not None:
            clauses.append("machine_id = ?")
            params.append(machine_id)
        query = "SELECT json FROM assessments"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        # created_at is always UTC ISO-8601 ("+00:00" with fixed microsecond
        # precision), so its lexicographic order equals chronological order;
        # assessment_id breaks ties deterministically.
        query += " ORDER BY created_at DESC, assessment_id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._lock:
            rows = self._db.execute(query, params).fetchall()
        return [Assessment.model_validate_json(row[0]) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._db.close()


__all__ = [
    "AssessmentAlreadyExists",
    "AssessmentStore",
    "DEFAULT_PAGE",
    "MAX_PAGE",
    "MIGRATIONS",
    "run_migrations",
]


# Guard against accidental API growth: append-only means insert/get/list/close only.
for _name in ("update", "replace", "delete", "upsert", "remove", "set"):
    if hasattr(AssessmentStore, _name):  # pragma: no cover - fails loudly at import
        raise RuntimeError(f"AssessmentStore must stay append-only (found '{_name}')")
del _name
