from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .common import Recommendation, RiskLevel

SensorParameter = Literal[
    "air_temperature_k", "process_temperature_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"
]


class SensorReading(BaseModel):
    """One snapshot of machine telemetry (AI4I 2020 schema)."""

    product_quality: Literal["L", "M", "H"] = Field("M", description="Product quality variant (AI4I 'Type')")
    air_temperature_k: float = Field(..., ge=250, le=350)
    process_temperature_k: float = Field(..., ge=250, le=400)
    rotational_speed_rpm: float = Field(..., gt=0, le=5000)
    torque_nm: float = Field(..., ge=0, le=200)
    tool_wear_min: float = Field(..., ge=0, le=500)

    model_config = {
        "json_schema_extra": {
            "example": {
                "product_quality": "L",
                "air_temperature_k": 302.4,
                "process_temperature_k": 310.6,
                "rotational_speed_rpm": 1352,
                "torque_nm": 54.2,
                "tool_wear_min": 212,
            }
        }
    }


class FailureModeScore(BaseModel):
    code: str
    name: str
    probability: float
    threshold: float
    reliable: bool = Field(True, description="False if the mode model's validation F1 is too low to raise flags")
    flagged: bool


class Contribution(BaseModel):
    feature: str
    label: str
    value: float
    contribution: float = Field(..., description="SHAP value in log-odds; > 0 pushes risk up")


class SensorAssessment(BaseModel):
    failure_probability: float
    risk_level: RiskLevel
    decision_threshold: float
    failure_modes: list[FailureModeScore]
    contributions: list[Contribution]
    base_value: float
    derived: dict[str, float]
    warnings: list[str]
    recommendations: list[Recommendation] = []


class BatchRequest(BaseModel):
    readings: list[SensorReading] = Field(..., min_length=1, max_length=1000)


class WhatIfRequest(BaseModel):
    reading: SensorReading
    parameter: SensorParameter
    values: list[float] | None = Field(None, max_length=200, description="Explicit values to try")
    start: float | None = None
    stop: float | None = None
    steps: int = Field(25, ge=2, le=200)

    @model_validator(mode="after")
    def _range_complete(self):
        if self.values is None and (self.start is None) != (self.stop is None):
            raise ValueError("Provide both start and stop, or neither (defaults to the training p1-p99 range)")
        return self


class WhatIfPoint(BaseModel):
    value: float
    failure_probability: float
    risk_level: RiskLevel
    top_mode: str | None
    top_mode_probability: float | None


class WhatIfResponse(BaseModel):
    parameter: SensorParameter
    baseline_value: float
    baseline_probability: float
    points: list[WhatIfPoint]
