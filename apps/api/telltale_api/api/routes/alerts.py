from fastapi import APIRouter, Depends, HTTPException

from ...schemas.fleet import Alert, AlertStatus, AlertUpdate
from ..deps import AppState, get_state

router = APIRouter()


@router.get("", response_model=list[Alert])
async def list_alerts(
    status: AlertStatus | None = None, machine_id: str | None = None, state: AppState = Depends(get_state)
) -> list[Alert]:
    return state.alerts.list(status=status, machine_id=machine_id)


def _set(alert_id: str, status: AlertStatus, body: AlertUpdate | None, state: AppState) -> Alert:
    alert = state.alerts.set_status(alert_id, status, body.note if body else None)
    if alert is None:
        raise HTTPException(404, "Alert not found")
    return alert


@router.post("/{alert_id}/acknowledge", response_model=Alert)
async def acknowledge(alert_id: str, body: AlertUpdate | None = None, state: AppState = Depends(get_state)) -> Alert:
    return _set(alert_id, "acknowledged", body, state)


@router.post("/{alert_id}/resolve", response_model=Alert)
async def resolve(alert_id: str, body: AlertUpdate | None = None, state: AppState = Depends(get_state)) -> Alert:
    return _set(alert_id, "resolved", body, state)
