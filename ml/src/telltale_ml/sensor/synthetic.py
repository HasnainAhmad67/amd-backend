"""AI4I-style synthetic data, following the generation rules published with AI4I 2020.

Used for (1) tests and offline smoke runs and (2) the API's live fleet simulator, so the
demo fleet behaves like the data the model was trained on. NOT a substitute for the real
dataset when reporting metrics.

Published AI4I failure rules (Matzka, 2020):
  TWF  tool wear between 200 and 240 min -> replaced or fails
  HDF  (process - air temperature) < 8.6 K  AND  rotational speed < 1380 rpm
  PWF  power < 3500 W  OR  power > 9000 W
  OSF  tool wear * torque > 11000 (L) / 12000 (M) / 13000 (H) Nm·min
  RNF  0.1 % random failures
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import RPM_TO_RAD_S

OSF_LIMITS = {"L": 11_000.0, "M": 12_000.0, "H": 13_000.0}


def failure_flags(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Apply the AI4I rules to raw readings. Returns TWF/HDF/PWF/OSF/RNF/machine_failure columns."""
    delta = df["process_temperature_k"] - df["air_temperature_k"]
    power = df["torque_nm"] * df["rotational_speed_rpm"] * RPM_TO_RAD_S
    strain = df["tool_wear_min"] * df["torque_nm"]
    limit = df["product_quality"].map(OSF_LIMITS)
    wear = df["tool_wear_min"]
    twf_p = np.clip((wear - 200.0) / 40.0, 0.0, 1.0) * 0.35
    out = pd.DataFrame(index=df.index)
    out["TWF"] = (rng.random(len(df)) < twf_p).astype(int)
    out["HDF"] = ((delta < 8.6) & (df["rotational_speed_rpm"] < 1380)).astype(int)
    out["PWF"] = ((power < 3500) | (power > 9000)).astype(int)
    out["OSF"] = (strain > limit).astype(int)
    out["RNF"] = (rng.random(len(df)) < 0.001).astype(int)
    out["machine_failure"] = out[["TWF", "HDF", "PWF", "OSF", "RNF"]].max(axis=1)
    return out


def generate(n: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """i.i.d. AI4I-like rows with labels (same columns as ``load_ai4i``)."""
    rng = np.random.default_rng(seed)
    quality = rng.choice(["L", "M", "H"], size=n, p=[0.6, 0.3, 0.1])
    air = 300.0 + rng.normal(0, 2.0, n)
    process = air + 10.0 + rng.normal(0, 1.0, n)
    torque = np.clip(rng.normal(40.0, 10.0, n), 3.0, None)
    rpm = np.clip(1540.0 - 12.0 * (torque - 40.0) + rng.normal(0, 80.0, n), 1150.0, 2900.0)
    wear = rng.uniform(0, 250, n).round()
    df = pd.DataFrame(
        {
            "product_quality": quality,
            "air_temperature_k": air.round(1),
            "process_temperature_k": process.round(1),
            "rotational_speed_rpm": rpm.round(),
            "torque_nm": torque.round(1),
            "tool_wear_min": wear,
        }
    )
    return pd.concat([df, failure_flags(df, rng)], axis=1)
