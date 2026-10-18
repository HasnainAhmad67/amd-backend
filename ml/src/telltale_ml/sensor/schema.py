"""Single source of truth for sensor field names, feature order and failure modes.

The API's Pydantic models, the training code and the frontend all use these snake_case names.
If you add a field, update: this file -> features.py -> API schemas/sensor.py -> web/src/api/types.ts.
"""

from __future__ import annotations

# Raw sensor inputs (what a machine / PLC would report).
SENSOR_FIELDS: tuple[str, ...] = (
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
)

# AI4I 2020 "Type" column: product quality variant (Low / Medium / High).
QUALITY_CODES: dict[str, int] = {"L": 0, "M": 1, "H": 2}

# AI4I CSV header -> canonical name.
AI4I_COLUMN_MAP: dict[str, str] = {
    "Type": "product_quality",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "TWF",
    "HDF": "HDF",
    "PWF": "PWF",
    "OSF": "OSF",
    "RNF": "RNF",
}

# Model input features, in the exact order the boosters were trained with.
FEATURES: tuple[str, ...] = (
    "quality_code",
    *SENSOR_FIELDS,
    "temp_delta_k",  # process - air temperature (heat dissipation headroom)
    "power_w",  # mechanical power = torque * angular velocity
    "strain_nm_min",  # tool wear * torque (overstrain indicator)
)

FEATURE_LABELS: dict[str, str] = {
    "quality_code": "Product quality (L/M/H)",
    "air_temperature_k": "Air temperature [K]",
    "process_temperature_k": "Process temperature [K]",
    "rotational_speed_rpm": "Rotational speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
    "temp_delta_k": "Process-air temperature gap [K]",
    "power_w": "Mechanical power [W]",
    "strain_nm_min": "Tool wear x torque [Nm·min]",
}

# RNF ("random failure") is excluded on purpose: it is random by construction in AI4I,
# so any model "predicting" it would be fitting noise.
FAILURE_MODES: dict[str, str] = {
    "TWF": "Tool wear failure",
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
}

TARGET = "machine_failure"
