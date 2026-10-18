"""Turns model outputs into prioritised inspection recommendations using data/playbook.yaml."""

from pathlib import Path
from typing import Any

import yaml

from ..schemas.common import Recommendation


class Playbook:
    def __init__(self, path: Path):
        with open(path, encoding="utf-8") as fh:
            self.data: dict[str, Any] = yaml.safe_load(fh)

    # -------------------------------------------------------------------- sensor
    def for_sensor(self, assessment: dict[str, Any]) -> list[Recommendation]:
        risk = assessment["risk_level"]
        modes = assessment["failure_modes"]
        recs: list[Recommendation] = []

        selected = [m for m in modes if m["flagged"]]
        if not selected and risk != "low" and modes:
            # Elevated overall risk: surface the most likely mechanism from a reliable mode model.
            selected = [m for m in modes if m.get("reliable", True)][:1]
        for m in selected:
            entry = self.data["failure_modes"].get(m["code"])
            if not entry:
                continue
            recs.append(
                Recommendation(
                    title=entry["title"],
                    reason=f"{m['name']} estimated at {m['probability']:.0%} (alert threshold {m['threshold']:.0%}).",
                    priority="high" if risk == "high" or m["flagged"] else "medium",
                    steps=entry["steps"],
                    source=f"failure_mode:{m['code']}",
                )
            )

        if risk != "low":
            drivers = [c for c in assessment["contributions"] if c["contribution"] > 0][:2]
            for c in drivers:
                text = self.data["drivers"].get(c["feature"])
                if text:
                    recs.append(
                        Recommendation(
                            title=f"Driver: {c['label']}",
                            reason=f"{text} (current value {c['value']:.1f}, +{c['contribution']:.2f} log-odds).",
                            priority="medium",
                            steps=[],
                            source=f"driver:{c['feature']}",
                        )
                    )

        if not recs:
            recs.append(
                Recommendation(
                    title="Operating normally",
                    reason=f"Estimated failure risk {assessment['failure_probability']:.1%} is in the low band.",
                    priority="low",
                    steps=["Continue routine monitoring and planned maintenance."],
                    source="baseline",
                )
            )
        return recs

    # -------------------------------------------------------------------- sound
    def for_acoustic(self, machine_type: str, analysis: dict[str, Any]) -> list[Recommendation]:
        if analysis["status"] == "normal":
            if analysis.get("p_value") is not None:
                why = f"{analysis['p_value']:.0%} of normal recordings sound at least this unusual - nothing stands out."
            else:
                why = f"Anomaly score {analysis['score']:.2f} is below {analysis['threshold']:.2f}."
            return [
                Recommendation(
                    title="Sound within normal range",
                    reason=why,
                    priority="low",
                    steps=["No action needed. Re-record periodically to keep a trend."],
                    source=f"acoustic:{machine_type}",
                )
            ]
        entry = self.data["acoustic"].get(machine_type) or self.data["acoustic"]["default"]
        band = self._loudest_deviation(analysis)
        if analysis.get("unusual_percentile") is not None:
            how = f"More unusual than {analysis['unusual_percentile']:.1f}% of normal recordings (p = {analysis['p_value']:.3f})"
        else:
            how = f"Anomaly score {analysis['score']:.2f} is {analysis['severity']:.1f}x the threshold"
        reason = (
            f"{how}; strongest deviation around {analysis['peak_time_s']:.1f}s"
            + (f", near {band:.0f} Hz." if band else ".")
        )
        return [
            Recommendation(
                title=entry["title"],
                reason=reason,
                priority="high" if analysis["status"] == "anomalous" else "medium",
                steps=entry["steps"],
                source=f"acoustic:{machine_type}",
            )
        ]

    @staticmethod
    def _loudest_deviation(analysis: dict[str, Any]) -> float | None:
        dev = analysis.get("band_deviation") or {}
        z, hz = dev.get("z") or [], dev.get("band_hz") or []
        if not z:
            return None
        i = max(range(len(z)), key=lambda k: abs(z[k]))
        return hz[i] if abs(z[i]) >= 2 else None

    # -------------------------------------------------------------------- bearing
    def for_bearing(self, diagnosis: dict[str, Any]) -> list[Recommendation]:
        cls = diagnosis["predicted_class"]
        entry = self.data["bearing"].get(cls)
        if entry is None:
            return [
                Recommendation(
                    title="Bearing sounds healthy",
                    reason=f"Classified as '{diagnosis['label']}' with {diagnosis['confidence']:.0%} confidence.",
                    priority="low",
                    steps=["No action needed."],
                    source=f"bearing:{cls}",
                )
            ]
        confident = diagnosis["confidence"] >= 0.6 and diagnosis["segment_agreement"] >= 0.6 and not diagnosis.get("uncertain")
        others = [lbl for c, lbl in zip(diagnosis.get("prediction_set", []), diagnosis.get("prediction_set_labels", []), strict=False) if c != cls]
        return [
            Recommendation(
                title=entry["title"],
                reason=(
                    f"Classified as '{diagnosis['label']}' with {diagnosis['confidence']:.0%} confidence "
                    f"({diagnosis['segment_agreement']:.0%} of segments agree)."
                    + (f" It could also be: {', '.join(others)}." if others else "")
                    + ("" if confident else " Low confidence: confirm with vibration analysis before acting.")
                ),
                priority="high" if confident else "medium",
                steps=entry["steps"],
                source=f"bearing:{cls}",
            )
        ]
