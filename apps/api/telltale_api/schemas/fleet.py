from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from .common import Health, RiskLevel, SoundStatus
from .sensor import SensorAssessment, SensorReading

AlertStatus = Literal["open", "acknowledged", "resolved"]
AlertSource = Literal["sensor", "acoustic"]
MaintenanceAction = Literal["replace_tool", "service_cooling", "reduce_load", "inspect_drive"]


class AcousticSnapshot(BaseModel):
    machine_type: str
    status: SoundStatus
    score: float
    severity: float
    analyzed_at: datetime
    filename: str | None = None


class HistoryPoint(BaseModel):
    t: datetime
    reading: SensorReading
    failure_probability: float | None
    risk_level: RiskLevel | None
    top_mode: str | None


class ToolLife(BaseModel):
    wear_min: float
    replace_at_min: float
    remaining_use_min: float  # minutes of cutting left before the tool-wear failure band
    eta_minutes: float | None  # wall-clock minutes until then at the current pace (None = not enough history)


class MachineSummary(BaseModel):
    id: str
    name: str
    line: str
    product_quality: str
    acoustic_profile: str | None
    simulated_scenario: str
    health: Health
    latest_reading: SensorReading | None
    failure_probability: float | None
    risk_level: RiskLevel | None
    top_mode: str | None
    acoustic: AcousticSnapshot | None
    open_alerts: int
    sparkline: list[float]
    tool_life: ToolLife | None = None


class MachineDetail(MachineSummary):
    assessment: SensorAssessment | None
    history: list[HistoryPoint]


class FleetOverview(BaseModel):
    machines: int
    by_health: dict[str, int]
    open_alerts: int
    models_online: dict[str, bool]
    simulator: bool


class Alert(BaseModel):
    id: str
    machine_id: str
    machine_name: str
    source: AlertSource
    severity: Literal["warning", "critical"]
    title: str
    detail: str
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    note: str | None = None


class AlertUpdate(BaseModel):
    note: str | None = None


class MaintenanceRequest(BaseModel):
    action: MaintenanceAction
    note: str | None = None


class SummaryResponse(BaseModel):
    machine_id: str
    summary: str
    source: str  # "llm" | "template"
