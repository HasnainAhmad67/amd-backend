"""Demo fleet: simulated machines streaming AI4I-style telemetry, scored live by the sensor model.

Replace ``FleetSimulator._next_reading`` with a real data source (MQTT, OPC UA, a CSV replay, ...)
to go beyond the demo — everything downstream (scoring, history, alerts, UI) stays the same.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..schemas.fleet import AcousticSnapshot, MaintenanceAction
from .alerts import AlertStore
from .registry import ModelRegistry

log = logging.getLogger("telltale.fleet")

CYCLE = 120  # ticks per degradation cycle (~6 min at 3 s/tick)
WEAR_PER_TICK = {"L": 2.0, "M": 3.0, "H": 5.0}  # AI4I: H/M/L add 5/3/2 min of tool wear per process
HEALTH_RANK = {"unknown": 0, "ok": 1, "watch": 2, "critical": 3}
RISK_TO_HEALTH = {"low": "ok", "medium": "watch", "high": "critical"}
SOUND_TO_HEALTH = {"normal": "ok", "watch": "watch", "anomalous": "critical"}


@dataclass
class MachineSpec:
    id: str
    name: str
    line: str
    product_quality: str = "M"
    acoustic_profile: str | None = None
    scenario: str = "healthy"


@dataclass
class MachineState:
    spec: MachineSpec
    air: float
    wear: float
    step: int
    load_relief: int = 0  # ticks of reduced load remaining after a "reduce_load" action
    history: deque = field(default_factory=lambda: deque(maxlen=240))
    reading: dict[str, Any] | None = None
    assessment: dict[str, Any] | None = None
    acoustic: AcousticSnapshot | None = None

    @property
    def health(self) -> str:
        levels = []
        if self.assessment:
            levels.append(RISK_TO_HEALTH[self.assessment["risk_level"]])
        if self.acoustic:
            levels.append(SOUND_TO_HEALTH[self.acoustic.status])
        return max(levels, key=HEALTH_RANK.__getitem__) if levels else "unknown"


# AI4I: tool-wear failures happen between 200 and 240 min of wear, so plan the change before 200.
TOOL_REPLACE_AT_MIN = 200.0


def tool_life(st: MachineState) -> dict[str, Any] | None:
    """Remaining tool life: minutes of cutting left before the wear band, and a wall-clock ETA at the current pace."""
    if not st.reading:
        return None
    wear = float(st.reading["tool_wear_min"])
    remaining = max(0.0, TOOL_REPLACE_AT_MIN - wear)
    # Pace = wear gained per real minute, measured only since the last tool change (wear dropping = new tool).
    since_change: list[tuple[datetime, float]] = []
    for h in list(st.history)[-30:]:
        w = float(h["reading"]["tool_wear_min"])
        if since_change and w < since_change[-1][1]:
            since_change = []
        since_change.append((h["t"], w))
    eta = None
    if len(since_change) >= 2:
        minutes = (since_change[-1][0] - since_change[0][0]).total_seconds() / 60
        gained = since_change[-1][1] - since_change[0][1]
        if minutes > 0 and gained > 0:
            eta = remaining / (gained / minutes)
    return {"wear_min": wear, "replace_at_min": TOOL_REPLACE_AT_MIN, "remaining_use_min": remaining, "eta_minutes": eta}


def load_specs(path: Path) -> list[MachineSpec]:
    with open(path, encoding="utf-8") as fh:
        return [MachineSpec(**m) for m in yaml.safe_load(fh)["machines"]]


class FleetSimulator:
    def __init__(self, specs: list[MachineSpec], registry: ModelRegistry, alerts: AlertStore, seed: int = 7):
        self.registry = registry
        self.alerts = alerts
        self.rng = np.random.default_rng(seed)
        self.machines: dict[str, MachineState] = {
            s.id: MachineState(
                spec=s,
                air=float(300 + self.rng.normal(0, 1)),
                wear=float(self.rng.uniform(0, 120)),
                step=int(self.rng.integers(0, CYCLE // 2)),
            )
            for s in specs
        }

    # ------------------------------------------------------------------ simulation
    def _next_reading(self, st: MachineState) -> dict[str, Any]:
        rng, spec = self.rng, st.spec
        st.step += 1
        phase = (st.step % CYCLE) / CYCLE  # 0 -> 1 ramp of the injected degradation

        st.air = float(np.clip(st.air + rng.normal(0, 0.15) + 0.02 * (300 - st.air), 295, 305))
        delta = 10.0 + rng.normal(0, 0.5)
        torque = rng.normal(40.0, 6.0)
        rpm_offset = 0.0

        if spec.scenario == "cooling_fault":
            delta -= 3.2 * phase
            rpm_offset -= 260 * phase
        elif spec.scenario == "overload":
            torque += 24 * phase
        elif spec.scenario == "power_anomaly" and phase > 0.55 and rng.random() < 0.5:
            torque += 22
            rpm_offset += 320

        if st.load_relief > 0:
            st.load_relief -= 1
            torque = min(torque, 38.0)

        torque = float(np.clip(torque, 5, 90))
        rpm = float(np.clip(1540 - 12 * (torque - 40) + rpm_offset + rng.normal(0, 50), 1150, 2900))

        st.wear += WEAR_PER_TICK.get(spec.product_quality, 3.0)
        replace_at = 250 if spec.scenario == "tool_wear" else 190  # healthy machines get preventive tool changes
        if st.wear >= replace_at:
            st.wear = 0.0

        return {
            "product_quality": spec.product_quality,
            "air_temperature_k": round(st.air, 1),
            "process_temperature_k": round(st.air + delta, 1),
            "rotational_speed_rpm": round(rpm),
            "torque_nm": round(torque, 1),
            "tool_wear_min": round(st.wear),
        }

    def tick(self) -> None:
        states = list(self.machines.values())
        readings = [self._next_reading(st) for st in states]
        predictor = self.registry.sensor
        assessments = predictor.assess(readings) if predictor else [None] * len(states)
        now = datetime.now(timezone.utc)
        for st, reading, assessment in zip(states, readings, assessments, strict=True):
            st.reading, st.assessment = reading, assessment
            top = assessment["failure_modes"][0]["code"] if assessment and assessment["failure_modes"] else None
            st.history.append(
                {
                    "t": now,
                    "reading": reading,
                    "failure_probability": assessment["failure_probability"] if assessment else None,
                    "risk_level": assessment["risk_level"] if assessment else None,
                    "top_mode": top,
                }
            )
            if assessment and assessment["risk_level"] == "high" and not self.alerts.active_for(st.spec.id, "sensor"):
                mode = assessment["failure_modes"][0] if assessment["failure_modes"] else None
                self.alerts.create(
                    machine_id=st.spec.id,
                    machine_name=st.spec.name,
                    source="sensor",
                    severity="critical",
                    title=f"High failure risk{': ' + mode['name'] if mode else ''}",
                    detail=f"Estimated failure probability {assessment['failure_probability']:.0%}.",
                )

    async def run(self, interval_s: float) -> None:
        log.info("Fleet simulator started (%d machines, %.1fs interval)", len(self.machines), interval_s)
        while True:
            try:
                self.tick()
            except Exception:  # keep the demo alive no matter what
                log.exception("Simulator tick failed")
            await asyncio.sleep(interval_s)

    # ------------------------------------------------------------------ operator actions
    def apply_maintenance(self, machine_id: str, action: MaintenanceAction, note: str | None) -> MachineState:
        st = self.machines[machine_id]
        if action == "replace_tool":
            st.wear = 0.0
        elif action == "service_cooling" or action == "inspect_drive":
            st.step = 0  # restart the degradation cycle
        elif action == "reduce_load":
            st.load_relief = 40
        self.alerts.resolve_for(machine_id, "sensor", note or f"Maintenance action: {action}")
        return st

    def record_acoustic(self, machine_id: str, analysis: dict[str, Any], filename: str | None) -> None:
        st = self.machines[machine_id]
        st.acoustic = AcousticSnapshot(
            machine_type=analysis["machine_type"],
            status=analysis["status"],
            score=analysis["score"],
            severity=analysis["severity"],
            analyzed_at=datetime.now(timezone.utc),
            filename=filename,
        )
        if analysis["status"] == "anomalous" and not self.alerts.active_for(machine_id, "acoustic"):
            self.alerts.create(
                machine_id=machine_id,
                machine_name=st.spec.name,
                source="acoustic",
                severity="critical" if analysis["severity"] >= 1.5 else "warning",
                title=f"Abnormal {analysis['machine_type']} sound",
                detail=(
                    f"More unusual than {analysis['unusual_percentile']:.1f}% of normal recordings."
                    if analysis.get("unusual_percentile") is not None
                    else f"Anomaly score {analysis['score']:.2f} ({analysis['severity']:.1f}x threshold)."
                ),
            )
