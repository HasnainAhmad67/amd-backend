from fastapi import APIRouter, Depends

from ..deps import AppState, get_state

router = APIRouter()


@router.get("")
def model_status(state: AppState = Depends(get_state)) -> dict:
    """Loaded models with provenance, evaluation metrics and caveats (for the Models page)."""
    return state.registry.status(detailed=True)


@router.post("/reload")
def reload_models(state: AppState = Depends(get_state)) -> dict:
    """Re-read TELLTALE_MODELS_DIR, e.g. after training while the API is running."""
    state.registry.load()
    return state.registry.status()
