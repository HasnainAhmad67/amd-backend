"""Central AI Analysis Agent - slice A template provider (deterministic, offline).

This module is the agent's only bridge between an immutable assessment snapshot and
an AI explanation. It:

- projects the snapshot into a facts-only payload (:func:`project_assessment`) -
  the SAME payload a future LLM provider would receive. Model identity fingerprints,
  filesystem paths, SQL, raw audio and any store internals are deliberately NOT part
  of the projection;
- renders a deterministic explanation from those facts and nothing else
  (:class:`TemplateExplainProvider`) - no network, no randomness, no clock reads
  beyond the stored timestamps;
- builds the deterministic refusal / fallback outputs used when evidence is
  insufficient or generation fails.

The provider NEVER computes, modifies or re-derives model outputs: every number it
prints is a stored value, every status it quotes is a stored status, and every
limitation is copied from the stored snapshot.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from ..schemas.ai import (
    MAX_EXPLANATION_CHARS,
    PROMPT_VERSION,
    AiExplainOutput,
    AiModalityNote,
)
from ..schemas.assessment import Assessment

# Documented instruction set for this slice. The template provider obeys it
# deterministically; a future provider receives the same text as its system prompt.
SYSTEM_INSTRUCTIONS = (
    "You explain an immutable predictive-maintenance assessment snapshot for a maintenance "
    "technician. Use ONLY the JSON facts provided. Quote stored values verbatim; never invent "
    "numbers, ids, timestamps, statuses or recommendations. Preserve uncertainty, missing "
    "evidence and model limitations. State clearly that results are model indications, not "
    "confirmed faults, and that missing evidence limits interpretation. Never claim "
    "machine-health completeness or coverage beyond the requested modalities."
)

_WORD = re.compile(r"\s+")
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?")


def _clean(text: Any, limit: int = MAX_EXPLANATION_CHARS, quotes: bool = True) -> str:
    """Flatten a stored string for display: single-spaced, bounded. Meaning is preserved.

    ``quotes=False`` is used for embedded PROSE (stored reasons, flag explanations,
    limitations) so an embedded double quote can never look like a quoted status
    token; the agent's own quoted stored values keep real double quotes, which the
    grounding validator then checks against the snapshot.
    """
    flat = _WORD.sub(" ", str(text)).strip()
    if not quotes:
        flat = flat.replace('"', "'")
    return flat[:limit]


def _num(value: Any) -> str:
    """Render a stored number with its exact shortest round-trip form (grounding-safe)."""
    if isinstance(value, bool):  # never treat booleans as numbers
        return "true" if value else "false"
    if isinstance(value, int):
        return repr(int(value))
    return repr(float(value))


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if value else None


def _render_evidence_value(value: Any) -> str:
    """Render one stored evidence value (numbers exactly; anything else as text)."""
    if isinstance(value, bool):  # never treat booleans as numbers
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _num(value)
    return _clean(value)


def project_assessment(assessment: Assessment) -> dict[str, Any]:
    """Facts-only projection of one immutable snapshot.

    This is the complete set of fields the agent may read (and the payload a future
    provider would receive). Fingerprints, paths, SQL and storage internals are
    deliberately excluded; only identity name/version/training dataset are included.
    """
    modalities: list[dict[str, Any]] = []
    for m in assessment.modalities:
        output = m.output.model_dump(mode="json") if m.output is not None else None
        modalities.append(
            {
                "kind": m.kind,
                "availability": m.availability,
                "reason_code": m.reason_code,
                "source_mode": m.source_mode,
                "observed_at": _iso(m.observed_at),
                "inference_at": _iso(m.inference_at),
                "model_name": m.model_identity.name if m.model_identity else None,
                "model_version": m.model_identity.version if m.model_identity else None,
                "training_dataset": m.model_identity.training_dataset if m.model_identity else None,
                "limitations": list(m.limitations),
                "output": output,
                "evidence": [
                    {
                        "evidence_id": str(e.evidence_id),
                        "kind": e.kind,
                        "value": e.value,
                        "unit": e.unit,
                        "source": e.source,
                    }
                    for e in assessment.evidence
                    if e.source == m.kind
                ],
            }
        )
    return {
        "machine_id": assessment.machine_id,
        "overall_status": assessment.overall_status,
        "status_reasons": list(assessment.status_reasons),
        "expected_modalities": list(assessment.expected_modalities),
        "evidence_completeness": assessment.evidence_completeness.model_dump(mode="json"),
        "quality_flags": [
            {"code": f.code, "severity": f.severity, "explanation": f.explanation}
            for f in assessment.quality_flags
        ],
        "modalities": modalities,
    }


def _modality_facts(kind: str, output: dict[str, Any]) -> list[str]:
    """Deterministic 'stored facts' sentences for one available modality output.

    Only fields that exist in the STORED typed output are quoted; nothing is derived.
    """
    facts: list[str] = []
    if kind == "sensor":
        facts.append(
            f'stored risk level "{output["risk_level"]}" with failure probability '
            f'{_num(output["failure_probability"])} (decision threshold {_num(output["decision_threshold"])})'
        )
        modes = output.get("failure_modes") or []
        if modes:
            top = modes[0]
            facts.append(
                f'most likely stored mechanism "{top["code"]}" (stored probability '
                f'{_num(top["probability"])}, threshold {_num(top["threshold"])}, flagged '
                f'{"true" if top["flagged"] else "false"})'
            )
    elif kind == "acoustic":
        facts.append(
            f'stored status "{output["status"]}" with anomaly score {_num(output["score"])} '
            f'against threshold {_num(output["threshold"])}'
        )
        if output.get("p_value") is not None:
            facts.append(
                f'conformal p-value {_num(output["p_value"])} (calibration '
                f'"{output["calibration"]}"), comparing with normal reference recordings only'
            )
        facts.append(f'{_num(output["duration_s"])} s of audio were scored by the model')
    elif kind == "bearing":
        facts.append(
            f'predicted class "{output["predicted_class"]}" with stored confidence '
            f'{_num(output["confidence"])}'
        )
        facts.append(
            f'conservative prediction set {list(output.get("prediction_set") or [])} '
            f'(coverage target {_num(output["coverage"])}); classifier uncertainty is '
            f'{"true" if output.get("uncertain") else "false"}'
        )
        if output.get("rpm_used") is not None:
            facts.append(
                f'shaft speed used {_num(output["rpm_used"])} rpm (default assumed: '
                f'{"true" if output.get("rpm_assumed") else "false"})'
            )
        facts.append(f'{_num(output["duration_s"])} s of audio were scored by the model')
    return facts


def _modality_note(mod: dict[str, Any]) -> tuple[AiModalityNote, list[str]]:
    lines: list[str] = []
    note = AiModalityNote(
        kind=mod["kind"],
        availability=mod["availability"],
        reason_code=mod["reason_code"],
        source_mode=mod["source_mode"],
        observed_at=mod["observed_at"],
        inference_at=mod["inference_at"],
        model_name=mod["model_name"],
        model_version=mod["model_version"],
        training_dataset=mod["training_dataset"],
        evidence_ids=[e["evidence_id"] for e in mod["evidence"]],
        facts=[],
    )
    if mod["availability"] == "available" and isinstance(mod["output"], dict):
        note.facts = _modality_facts(mod["kind"], mod["output"])
        lines.append(
            f'{mod["kind"].capitalize()}: availability "{mod["availability"]}" '
            f'(reason "{mod["reason_code"]}"), source mode "{mod["source_mode"]}"; '
            + ("; ".join(note.facts)) + "."
        )
    else:
        lines.append(
            f'{mod["kind"].capitalize()}: availability "{mod["availability"]}" '
            f'(reason "{mod["reason_code"]}"); no model output exists for this modality, '
            "so nothing is interpreted from it."
        )
    return note, lines


class TemplateExplainProvider:
    """Deterministic, offline explanation provider. No network, no randomness."""

    provider = "template"
    model: str | None = None

    def explain(self, assessment: Assessment) -> AiExplainOutput:
        facts = project_assessment(assessment)
        completeness = facts["evidence_completeness"]
        expected = facts["expected_modalities"]
        present = [m["kind"] for m in facts["modalities"] if m["availability"] == "available"]

        per_modality: list[AiModalityNote] = []
        lines: list[str] = []
        for mod in facts["modalities"]:
            note, mod_lines = _modality_note(mod)
            per_modality.append(note)
            lines.extend(mod_lines)

        for mod in facts["modalities"]:
            if mod["availability"] != "available":
                continue
            observed = mod["observed_at"] or "missing measurement timestamp"
            inferred = mod["inference_at"] or "no recorded inference time"
            identity_bits = [mod["model_name"] or "unnamed model", mod["model_version"] or "no version"]
            dataset = mod["training_dataset"] or "no training dataset recorded"
            evidence_bits = ", ".join(
                f'{e["kind"]} = {_render_evidence_value(e["value"])}' for e in mod["evidence"]
            )
            lines.append(
                f'{mod["kind"].capitalize()} evidence: observed at {observed}; inferred at {inferred}; '
                f'model {identity_bits[0]} version {identity_bits[1]}, trained on {dataset}. '
                f'Stored evidence values: {evidence_bits or "none recorded"}.'
            )

        flag_lines = []
        for flag in facts["quality_flags"]:
            detail = _clean(flag["explanation"], 400, quotes=False)
            flag_lines.append(f'"{flag["code"]}" ({flag["severity"]}) - {detail}')

        limitations: list[str] = []
        for mod in facts["modalities"]:
            limitations.extend(_clean(item, 600, quotes=False) for item in mod["limitations"])

        uncertainty_notes: list[str] = []
        if not present:
            uncertainty_notes.append(
                "No modality produced available evidence; the snapshot cannot be interpreted."
            )
        if any(m["observed_at"] is None for m in facts["modalities"] if m["availability"] == "available"):
            uncertainty_notes.append(
                "A measurement timestamp is missing; no age or freshness judgement is made."
            )
        uncertain_codes = {"BEARING_UNCERTAIN", "EVIDENCE_INCOMPLETE", "NO_AVAILABLE_MODALITY"}
        if any(f["code"] in uncertain_codes for f in facts["quality_flags"]):
            uncertainty_notes.append(
                "Conservative uncertainty flags are stored; a review priority is reported, "
                "not a confirmed fault."
            )
        if any(m["source_mode"] in ("dataset_demo", "simulated") for m in facts["modalities"]):
            uncertainty_notes.append(
                "At least one input is illustrative demo/simulated data, not a real-machine measurement."
            )
        if any(m["source_mode"] == "uploaded" for m in facts["modalities"]):
            uncertainty_notes.append(
                "Uploaded recordings are identified by hash only; conformal validity depends on "
                "resembling the calibration recordings."
            )

        reasons_part = ", ".join(f'"{_clean(r, 200, quotes=False)}"' for r in facts["status_reasons"])
        header = (
            f'Explanation of assessment {assessment.assessment_id} for machine {facts["machine_id"]}. '
            f'Stored overall status is "{facts["overall_status"]}" '
            f"(stored reasons: {reasons_part or 'none recorded'})."
        )
        completeness_note = (
            f'Requested modalities: {", ".join(expected)}; {completeness["present"]} of '
            f'{completeness["expected"]} requested modalities produced available evidence. '
            "This is the requested multimodal scope only: it is not machine-health completeness "
            "or machine coverage."
        )
        sections = [
            header,
            completeness_note,
            "Uncertainty and conservatism: watch, unknown and anomalous statuses are conservative "
            "review priorities, not confirmed faults; missing or uncertain evidence limits "
            "interpretation. Results are model indications, not confirmed faults, and this "
            "explanation quotes stored values only.",
            *lines,
            "Stored quality flags: " + ("; ".join(flag_lines) if flag_lines else "none stored."),
            "Model limitations preserved from the artifacts: "
            + ("; ".join(f"- {item}" for item in limitations) if limitations else "none stored."),
        ]
        text = _clean(" ".join(sections))[:MAX_EXPLANATION_CHARS]

        return AiExplainOutput(
            assessment_id=str(assessment.assessment_id),
            machine_id=assessment.machine_id,
            overall_status=assessment.overall_status,
            status_reasons=list(assessment.status_reasons),
            expected_modalities=list(assessment.expected_modalities),
            present_modalities=present,
            completeness_note=completeness_note,
            uncertainty_notes=uncertaintynotes(uncertainty_notes),
            per_modality=per_modality,
            quality_flag_codes=[f["code"] for f in facts["quality_flags"]],
            limitations=limitations,
            explanation_text=text,
            referenced_assessment_ids=[str(assessment.assessment_id)],
            referenced_evidence_ids=[eid for note in per_modality for eid in note.evidence_ids],
        )

    def prompt_sha256(self, assessment: Assessment) -> str:
        """Deterministic digest of the effective prompt (version + instructions + facts).
        The raw prompt itself is never stored."""
        canonical = json.dumps(
            {"prompt_version": PROMPT_VERSION, "instructions": SYSTEM_INSTRUCTIONS,
             "facts": project_assessment(assessment)},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def uncertaintynotes(notes: list[str]) -> list[str]:
    """Bounded, de-duplicated uncertainty notes (deterministic order)."""
    seen: list[str] = []
    for note in notes:
        flat = _clean(note, 300)
        if flat not in seen:
            seen.append(flat)
    return seen[:12]


def refusal_output(assessment: Assessment) -> AiExplainOutput:
    """Deterministic refusal: not enough available evidence to explain anything."""
    return AiExplainOutput(
        assessment_id=str(assessment.assessment_id),
        machine_id=assessment.machine_id,
        overall_status=assessment.overall_status,
        status_reasons=list(assessment.status_reasons),
        expected_modalities=list(assessment.expected_modalities),
        present_modalities=[m.kind for m in assessment.modalities if m.availability == "available"],
        completeness_note=(
            "No modality produced available evidence, so this assessment cannot be explained; "
            "the requested scope is recorded but no interpretation is offered."
        ),
        uncertainty_notes=["No available evidence was stored in this snapshot."],
        per_modality=[],
        quality_flag_codes=[f.code for f in assessment.quality_flags],
        limitations=[
            _clean(item, 600, quotes=False) for m in assessment.modalities for item in m.limitations
        ],
        explanation_text=(
            f"Assessment {assessment.assessment_id} for machine {assessment.machine_id} does not "
            f'contain enough available evidence to explain it. The stored overall status is '
            f'"{assessment.overall_status}" and no further interpretation is offered, because '
            "missing evidence limits interpretation; results remain model indications, not "
            "confirmed faults."
        ),
        referenced_assessment_ids=[str(assessment.assessment_id)],
        referenced_evidence_ids=[],
    )


def fallback_output(assessment: Assessment) -> AiExplainOutput:
    """Deterministic fallback: generation or validation failed; say only stored facts."""
    return AiExplainOutput(
        assessment_id=str(assessment.assessment_id),
        machine_id=assessment.machine_id,
        overall_status=assessment.overall_status,
        status_reasons=list(assessment.status_reasons),
        expected_modalities=list(assessment.expected_modalities),
        present_modalities=[m.kind for m in assessment.modalities if m.availability == "available"],
        completeness_note=(
            "A full explanation could not be produced for this assessment; only the stored "
            "status is reported and no interpretation beyond it is offered."
        ),
        uncertainty_notes=["The explanation template did not validate; no interpretation is offered."],
        per_modality=[],
        quality_flag_codes=[f.code for f in assessment.quality_flags],
        limitations=[
            _clean(item, 600, quotes=False) for m in assessment.modalities for item in m.limitations
        ],
        explanation_text=(
            f"Assessment {assessment.assessment_id} for machine {assessment.machine_id} has stored "
            f'overall status "{assessment.overall_status}". A full explanation could not be '
            "produced, so no additional interpretation is offered; missing or uncertain evidence "
            "limits interpretation, and results remain model indications, not confirmed faults."
        ),
        referenced_assessment_ids=[str(assessment.assessment_id)],
        referenced_evidence_ids=[],
    )
