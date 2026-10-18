"""Loading the AI4I 2020 predictive-maintenance dataset (UCI #601)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import raw_dir
from .schema import AI4I_COLUMN_MAP


def default_ai4i_path() -> Path:
    return raw_dir() / "ai4i2020" / "ai4i2020.csv"


def load_ai4i(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else default_ai4i_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"AI4I CSV not found at {path}. Run `telltale-download ai4i` (or `make data-sensor`)."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in AI4I_COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected AI4I columns: {missing}")
    return df.rename(columns=AI4I_COLUMN_MAP)[list(AI4I_COLUMN_MAP.values())]
