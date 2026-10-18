"""Feature engineering shared by training and serving. Never duplicate this logic elsewhere."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import pandas as pd

from .schema import FEATURES, QUALITY_CODES, SENSOR_FIELDS

RPM_TO_RAD_S = 2.0 * math.pi / 60.0


def readings_to_frame(readings: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Convert API-style dicts (snake_case keys) to a raw DataFrame."""
    df = pd.DataFrame(list(readings))
    missing = [c for c in ("product_quality", *SENSOR_FIELDS) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing sensor fields: {missing}")
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Raw readings (product_quality + SENSOR_FIELDS) -> model feature matrix in FEATURES order."""
    out = pd.DataFrame(index=df.index)
    quality = df["product_quality"].astype(str).str.upper().map(QUALITY_CODES)
    if quality.isna().any():
        bad = sorted(df.loc[quality.isna(), "product_quality"].astype(str).unique())
        raise ValueError(f"Unknown product_quality values {bad}; expected one of {list(QUALITY_CODES)}")
    out["quality_code"] = quality.astype(int)
    for field in SENSOR_FIELDS:
        out[field] = pd.to_numeric(df[field], errors="raise").astype(float)
    out["temp_delta_k"] = out["process_temperature_k"] - out["air_temperature_k"]
    out["power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * RPM_TO_RAD_S
    out["strain_nm_min"] = out["tool_wear_min"] * out["torque_nm"]
    return out[list(FEATURES)]
