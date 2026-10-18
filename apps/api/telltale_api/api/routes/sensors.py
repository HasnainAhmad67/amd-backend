from fastapi import APIRouter, Depends, HTTPException

from ...schemas.sensor import (
    BatchRequest,
    SensorAssessment,
    SensorReading,
    WhatIfRequest,
    WhatIfResponse,
)
from ..deps import AppState, get_state

router = APIRouter()


@router.get("/schema")
def sensor_schema(state: AppState = Depends(get_state)) -> dict:
    """Feature ranges seen in training (used by the UI to size what-if sliders)."""
    predictor = state.registry.require_sensor()
    meta = predictor.metadata
    return {
        "sensor_fields": meta["sensor_fields"],
        "feature_labels": meta["feature_labels"],
        "feature_ranges": meta["feature_ranges"],
        "risk_thresholds": meta["risk_thresholds"],
    }


@router.post("/predict", response_model=SensorAssessment)
def predict(reading: SensorReading, state: AppState = Depends(get_state)) -> dict:
    """Failure probability, risk band, failure-mode probabilities, SHAP contributions and recommended steps."""
    [result] = state.registry.require_sensor().assess([reading.model_dump()])
    result["recommendations"] = state.playbook.for_sensor(result)
    return result


@router.post("/predict/batch", response_model=list[SensorAssessment])
def predict_batch(body: BatchRequest, state: AppState = Depends(get_state)) -> list[dict]:
    results = state.registry.require_sensor().assess([r.model_dump() for r in body.readings])
    for r in results:
        r["recommendations"] = state.playbook.for_sensor(r)
    return results


@router.post("/what-if", response_model=WhatIfResponse)
def what_if(body: WhatIfRequest, state: AppState = Depends(get_state)) -> dict:
    """Sweep one sensor value while holding the rest fixed and return the model's risk curve."""
    predictor = state.registry.require_sensor()
    reading = body.reading.model_dump()
    if body.values is not None:
        values = body.values
    elif body.start is not None:
        step = (body.stop - body.start) / (body.steps - 1)
        values = [body.start + i * step for i in range(body.steps)]
    else:
        values = predictor.default_sweep(body.parameter, body.steps).tolist()
    try:
        points = predictor.what_if(reading, body.parameter, values)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    [baseline] = predictor.assess([reading])
    return {
        "parameter": body.parameter,
        "baseline_value": reading[body.parameter],
        "baseline_probability": baseline["failure_probability"],
        "points": points,
    }
