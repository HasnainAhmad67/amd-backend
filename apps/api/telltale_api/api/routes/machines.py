from fastapi import APIRouter, Depends, HTTPException

from ...schemas.fleet import FleetOverview, MachineDetail, MachineSummary, MaintenanceRequest, SummaryResponse
from ...services.fleet import MachineState, tool_life
from ..deps import AppState, get_state

router = APIRouter()


def _summary(st: MachineState, state: AppState) -> dict:
    a = st.assessment
    return {
        "id": st.spec.id,
        "name": st.spec.name,
        "line": st.spec.line,
        "product_quality": st.spec.product_quality,
        "acoustic_profile": st.spec.acoustic_profile,
        "simulated_scenario": st.spec.scenario,
        "health": st.health,
        "latest_reading": st.reading,
        "failure_probability": a["failure_probability"] if a else None,
        "risk_level": a["risk_level"] if a else None,
        "top_mode": a["failure_modes"][0]["code"] if a and a["failure_modes"] else None,
        "acoustic": st.acoustic,
        "open_alerts": state.alerts.count_active(st.spec.id),
        "sparkline": [h["failure_probability"] for h in list(st.history)[-60:] if h["failure_probability"] is not None],
        "tool_life": tool_life(st),
    }


def _get(machine_id: str, state: AppState) -> MachineState:
    st = state.fleet.machines.get(machine_id)
    if st is None:
        raise HTTPException(404, f"Unknown machine '{machine_id}'")
    return st


@router.get("/overview", response_model=FleetOverview)
async def overview(state: AppState = Depends(get_state)) -> dict:
    by_health: dict[str, int] = {"ok": 0, "watch": 0, "critical": 0, "unknown": 0}
    for st in state.fleet.machines.values():
        by_health[st.health] += 1
    return {
        "machines": len(state.fleet.machines),
        "by_health": by_health,
        "open_alerts": state.alerts.count_active(),
        "models_online": state.registry.online(),
        "simulator": state.settings.simulator_enabled,
    }


@router.get("", response_model=list[MachineSummary])
async def list_machines(state: AppState = Depends(get_state)) -> list[dict]:
    return [_summary(st, state) for st in state.fleet.machines.values()]


@router.get("/{machine_id}", response_model=MachineDetail)
async def machine_detail(machine_id: str, state: AppState = Depends(get_state)) -> dict:
    st = _get(machine_id, state)
    assessment = None
    if st.assessment:
        assessment = {**st.assessment, "recommendations": state.playbook.for_sensor(st.assessment)}
    return {**_summary(st, state), "assessment": assessment, "history": list(st.history)}


@router.post("/{machine_id}/maintenance", response_model=MachineSummary)
async def maintenance(machine_id: str, body: MaintenanceRequest, state: AppState = Depends(get_state)) -> dict:
    """Log an operator action. In the demo it also changes the simulated machine (e.g. new tool -> wear 0)."""
    _get(machine_id, state)
    st = state.fleet.apply_maintenance(machine_id, body.action, body.note)
    return _summary(st, state)


@router.post("/{machine_id}/summary", response_model=SummaryResponse)
async def summary(machine_id: str, state: AppState = Depends(get_state)) -> dict:
    """Plain-language briefing (template, or an LLM if TELLTALE_LLM_BASE_URL is set)."""
    st = _get(machine_id, state)
    facts: dict = {"machine": st.spec.name, "line": st.spec.line}
    if st.assessment:
        a = st.assessment
        recs = state.playbook.for_sensor(a)
        facts["sensor"] = {
            "failure_probability": round(a["failure_probability"], 3),
            "risk_level": a["risk_level"],
            "top_mode": a["failure_modes"][0]["name"] if a["failure_modes"] else None,
            "drivers": [c["label"] for c in a["contributions"] if c["contribution"] > 0][:3],
            "reading": st.reading,
        }
        facts["actions"] = [r.title for r in recs if r.priority != "low"]
    if st.acoustic:
        facts["acoustic"] = st.acoustic.model_dump(mode="json")
    text, source = await state.narrator.summarize(facts)
    return {"machine_id": machine_id, "summary": text, "source": source}
