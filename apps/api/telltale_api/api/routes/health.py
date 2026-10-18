from fastapi import APIRouter, Depends

from ... import __version__
from ..deps import AppState, get_state

router = APIRouter()


@router.get("/health")
def health(state: AppState = Depends(get_state)) -> dict:
    """Liveness + which models are loaded. Always 200 if the process is up."""
    return {
        "status": "ok",
        "version": __version__,
        "models": state.registry.online(),
        "acoustic_machine_types": sorted(state.registry.acoustic),
        "simulator": state.settings.simulator_enabled,
        "llm": state.narrator.llm_enabled,
        "sensor_bootstrap": {"status": state.bootstrap.status, "message": state.bootstrap.message},
    }
