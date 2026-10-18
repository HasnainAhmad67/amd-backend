"""Inference for the sensor failure-risk models. Torch-free; used directly by the API."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from ..utils import read_json
from .features import engineer, readings_to_frame
from .schema import FAILURE_MODES, FEATURE_LABELS, SENSOR_FIELDS


class SensorPredictor:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.metadata: dict[str, Any] = read_json(self.model_dir / "metadata.json")
        models = self.metadata["models"]
        self.features: list[str] = self.metadata["features"]
        self.failure_model = lgb.Booster(model_file=str(self.model_dir / models["failure"]["file"]))
        self.decision_threshold: float = float(models["failure"]["threshold"])
        self.mode_models = {
            code: (lgb.Booster(model_file=str(self.model_dir / m["file"])), float(m["threshold"]))
            for code, m in models["modes"].items()
        }
        # Older artifacts have no "reliable" key: treat them as reliable.
        self.mode_reliable = {code: bool(m.get("reliable", True)) for code, m in models["modes"].items()}
        self.risk_thresholds: dict[str, float] = self.metadata["risk_thresholds"]
        self.feature_ranges: dict[str, dict[str, float]] = self.metadata["feature_ranges"]

    # ------------------------------------------------------------------ helpers
    def risk_level(self, probability: float) -> str:
        if probability >= self.risk_thresholds["high"]:
            return "high"
        if probability >= self.risk_thresholds["medium"]:
            return "medium"
        return "low"

    def _range_warnings(self, row: Mapping[str, float]) -> list[str]:
        warnings = []
        for feature, value in row.items():
            rng = self.feature_ranges.get(feature)
            if rng and not (rng["min"] <= value <= rng["max"]):
                warnings.append(
                    f"{FEATURE_LABELS.get(feature, feature)} = {value:.1f} is outside the training range "
                    f"[{rng['min']:.1f}, {rng['max']:.1f}]; the estimate is an extrapolation."
                )
        return warnings

    # ------------------------------------------------------------------ public API
    def assess(self, readings: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        X = engineer(readings_to_frame(readings))
        proba = self.failure_model.predict(X)
        contrib = self.failure_model.predict(X, pred_contrib=True)  # (n, n_features + 1), log-odds
        mode_proba = {code: model.predict(X) for code, (model, _) in self.mode_models.items()}

        results = []
        for i in range(len(X)):
            row = X.iloc[i]
            p = float(proba[i])
            modes = sorted(
                (
                    {
                        "code": code,
                        "name": FAILURE_MODES.get(code, code),
                        "probability": float(mode_proba[code][i]),
                        "threshold": thr,
                        "reliable": self.mode_reliable[code],
                        "flagged": bool(self.mode_reliable[code] and mode_proba[code][i] >= thr),
                    }
                    for code, (_, thr) in self.mode_models.items()
                ),
                # Reliable modes first, so "most likely mechanism" never comes from a weak model.
                key=lambda m: (m["reliable"], m["probability"]),
                reverse=True,
            )
            contributions = sorted(
                (
                    {
                        "feature": f,
                        "label": FEATURE_LABELS.get(f, f),
                        "value": float(row[f]),
                        "contribution": float(contrib[i, j]),
                    }
                    for j, f in enumerate(self.features)
                ),
                key=lambda c: abs(c["contribution"]),
                reverse=True,
            )
            results.append(
                {
                    "failure_probability": p,
                    "risk_level": self.risk_level(p),
                    "decision_threshold": self.decision_threshold,
                    "failure_modes": modes,
                    "contributions": contributions,
                    "base_value": float(contrib[i, -1]),
                    "derived": {k: float(row[k]) for k in ("temp_delta_k", "power_w", "strain_nm_min")},
                    "warnings": self._range_warnings(row.to_dict()),
                }
            )
        return results

    def what_if(self, reading: Mapping[str, Any], parameter: str, values: Sequence[float]) -> list[dict[str, Any]]:
        """Sweep one raw sensor field while holding the others fixed."""
        if parameter not in SENSOR_FIELDS:
            raise ValueError(f"parameter must be one of {SENSOR_FIELDS}")
        variants = [{**reading, parameter: float(v)} for v in values]
        out = []
        for v, res in zip(values, self.assess(variants), strict=True):
            top = res["failure_modes"][0] if res["failure_modes"] else None
            out.append(
                {
                    "value": float(v),
                    "failure_probability": res["failure_probability"],
                    "risk_level": res["risk_level"],
                    "top_mode": top["code"] if top else None,
                    "top_mode_probability": top["probability"] if top else None,
                }
            )
        return out

    def default_sweep(self, parameter: str, steps: int = 25) -> np.ndarray:
        rng = self.feature_ranges[parameter]
        return np.linspace(rng["p01"], rng["p99"], steps)
