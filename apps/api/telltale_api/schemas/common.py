from typing import Literal

from pydantic import BaseModel

Priority = Literal["high", "medium", "low"]
RiskLevel = Literal["low", "medium", "high"]
SoundStatus = Literal["normal", "watch", "anomalous"]
Health = Literal["ok", "watch", "critical", "unknown"]


class Recommendation(BaseModel):
    title: str
    reason: str
    priority: Priority
    steps: list[str]
    source: str  # e.g. "failure_mode:HDF", "driver:torque_nm", "acoustic:fan", "bearing:outer_race"
