"""Alert stores: in-memory (AlertStore) and SQLite-backed (SqliteAlertStore, the default) with the same methods."""

import sqlite3
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from ..schemas.fleet import Alert, AlertSource, AlertStatus

MAX_ALERTS = 500


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AlertStore:
    def __init__(self) -> None:
        self._alerts: OrderedDict[str, Alert] = OrderedDict()

    def create(self, *, machine_id: str, machine_name: str, source: AlertSource, severity: str, title: str, detail: str) -> Alert:
        now = _now()
        alert = Alert(
            id=uuid.uuid4().hex[:12],
            machine_id=machine_id,
            machine_name=machine_name,
            source=source,
            severity=severity,
            title=title,
            detail=detail,
            status="open",
            created_at=now,
            updated_at=now,
        )
        self._alerts[alert.id] = alert
        while len(self._alerts) > MAX_ALERTS:
            self._alerts.popitem(last=False)
        return alert

    def active_for(self, machine_id: str, source: AlertSource) -> Alert | None:
        for alert in reversed(self._alerts.values()):
            if alert.machine_id == machine_id and alert.source == source and alert.status != "resolved":
                return alert
        return None

    def list(self, status: AlertStatus | None = None, machine_id: str | None = None) -> list[Alert]:
        items = [
            a
            for a in self._alerts.values()
            if (status is None or a.status == status) and (machine_id is None or a.machine_id == machine_id)
        ]
        return sorted(items, key=lambda a: a.created_at, reverse=True)

    def get(self, alert_id: str) -> Alert | None:
        return self._alerts.get(alert_id)

    def set_status(self, alert_id: str, status: AlertStatus, note: str | None = None) -> Alert | None:
        alert = self._alerts.get(alert_id)
        if alert is None:
            return None
        alert.status = status
        alert.updated_at = _now()
        if note:
            alert.note = note
        return alert

    def resolve_for(self, machine_id: str, source: AlertSource, note: str) -> int:
        n = 0
        for alert in self._alerts.values():
            if alert.machine_id == machine_id and alert.source == source and alert.status != "resolved":
                self.set_status(alert.id, "resolved", note)
                n += 1
        return n

    def count_active(self, machine_id: str | None = None) -> int:
        return sum(1 for a in self._alerts.values() if a.status != "resolved" and (machine_id is None or a.machine_id == machine_id))


class SqliteAlertStore(AlertStore):
    """Same interface as AlertStore, but alerts survive restarts (stored in a small SQLite file)."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("CREATE TABLE IF NOT EXISTS alerts (id TEXT PRIMARY KEY, created_at TEXT, json TEXT NOT NULL)")
            rows = self._db.execute("SELECT json FROM alerts ORDER BY created_at DESC LIMIT ?", (MAX_ALERTS,)).fetchall()
        for (raw,) in reversed(rows):
            alert = Alert.model_validate_json(raw)
            self._alerts[alert.id] = alert

    def _save(self, alert: Alert) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO alerts (id, created_at, json) VALUES (?, ?, ?)",
                (alert.id, alert.created_at.isoformat(), alert.model_dump_json()),
            )
            self._db.commit()

    def create(self, **kwargs) -> Alert:  # type: ignore[override]
        alert = super().create(**kwargs)
        self._save(alert)
        return alert

    def set_status(self, alert_id: str, status: AlertStatus, note: str | None = None) -> Alert | None:
        alert = super().set_status(alert_id, status, note)
        if alert is not None:
            self._save(alert)
        return alert
