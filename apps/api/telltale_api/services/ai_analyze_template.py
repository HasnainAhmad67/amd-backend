"""Central AI Analysis Agent - slice C: Multi-Model Analysis (deterministic, offline).

Describes how the AVAILABLE modality evidences of one immutable snapshot relate,
using fixed conservative templates and stored indication tokens only. No score,
p-value, confidence, status, id or timestamp is created, and no fused failure
probability, health score or risk score is ever computed: this is co-occurrence
description, not combination.
"""

from __future__ import annotations

import hashlib
import json

from ..schemas.ai import (
    MAX_EXPLANATION_CHARS,
    PROMPT_VERSION_ANALYZE,
    AgreementValue,
    AiAnalyzeOutput,
    AiModalitySummary,
    AiPairRelation,
)
from ..schemas.assessment import Assessment
from .ai_template import SYSTEM_INSTRUCTIONS, _clean, project_assessment

# Fixed non-claims, identical on every analysis (structured field + text footer).
NON_CLAIMS = [
    "This analysis is not a fault diagnosis and does not confirm any fault.",
    "This analysis is not machine-health completeness and makes no coverage claim.",
    "Missing or uncertain evidence limits interpretation; nothing was fabricated to fill gaps.",
    "No fused failure probability, health score or risk score is created here; stored model "
    "values are quoted only.",
]

FOOTER = (
    "This analysis describes co-occurrence of stored indications only. It is not a fault "
    "diagnosis, not machine-health completeness, and not coverage, and it creates no fused "
    "failure probability, health score or risk score."
)

REQUIRED_PHRASES = (
    "not a fault diagnosis",
    "not machine-health completeness",
    "not coverage",
    "no fused failure probability",
)

# Fixed conservative meanings for STORED overall statuses.
STATUS_MEANING = {
    "normal": 'The stored overall status is "{status}": no model-indicated review priority is recorded.',
    "watch": 'The stored overall status is "{status}": a conservative review priority, '
             "not a confirmed fault.",
    "critical": 'The stored overall status is "{status}": a model-indicated review priority, '
                "not a confirmed fault.",
    "unknown": 'The stored overall status is "{status}": evidence is incomplete, ambiguous or '
               "missing, so no normal interpretation is offered.",
}

CANONICAL_KINDS = ("sensor", "acoustic", "bearing")

BOTH_INDICATE = (
    "both modalities store a review priority; their stored indications co-occur, which is "
    "consistent stored evidence, not a confirmed fault"
)
ONE_INDICATES = (
    "one modality stores a review priority while the other stores none; the sources are "
    "unrelated, so no modality explains another and no causality is inferred"
)
BOTH_CLEAR = (
    "neither modality stores a review priority; their stored non-indications co-occur, which "
    "is still not a confirmation of health"
)
ANOMALY_WITH_UNCERTAINTY = (
    "an acoustic anomaly indication coexists with stored bearing classifier uncertainty; "
    "neither confirms nor explains the other"
)
UNCERTAINTY_LIMITS = (
    "stored classifier uncertainty in the bearing modality limits pairwise interpretation with "
    "the other modality; the co-occurrence is indeterminate"
)


def indication(kind: str, output: dict) -> tuple[str, str]:
    """Stored-only indication token for one AVAILABLE modality.

    Returns (class, phrase) where class is one of "indicates" / "clear" / "uncertain".
    """
    if kind == "sensor":
        risk = str(output.get("risk_level", ""))
        if risk in ("medium", "high"):
            return "indicates", f'review priority stored (stored risk level "{risk}")'
        return "clear", f'no stored review priority (stored risk level "{risk}")'
    if kind == "acoustic":
        status = str(output.get("status", ""))
        if status in ("watch", "anomalous"):
            return "indicates", f'anomaly indication stored (stored status "{status}")'
        return "clear", f'no stored anomaly (stored status "{status}")'
    # bearing
    if bool(output.get("uncertain")):
        return "uncertain", "classifier uncertainty stored (stored uncertain flag true)"
    predicted = str(output.get("predicted_class", ""))
    classes = [str(c) for c in (output.get("prediction_set") or [])]
    if classes and all(c == "normal" for c in classes):
        return "clear", "no stored review priority (conservative set contains only the normal class)"
    return (
        "indicates",
        f'fault classes present in the conservative set (stored predicted class "{predicted}")',
    )


def _pair_relation(kind_a: str, class_a: str, kind_b: str, class_b: str) -> str:
    if "uncertain" in (class_a, class_b):
        if {kind_a, kind_b} == {"acoustic", "bearing"}:
            if (kind_a == "acoustic" and class_a == "indicates") or (
                kind_b == "acoustic" and class_b == "indicates"
            ):
                return ANOMALY_WITH_UNCERTAINTY
        return UNCERTAINTY_LIMITS
    if class_a == "indicates" and class_b == "indicates":
        return BOTH_INDICATE
    if class_a == "clear" and class_b == "clear":
        return BOTH_CLEAR
    return ONE_INDICATES


def _agreement(classes: list[str]) -> tuple[AgreementValue, str]:
    if "uncertain" in classes:
        return "indeterminate", (
            "at least one available modality stores classifier uncertainty, so agreement between "
            "modalities is not asserted"
        )
    if all(c == "clear" for c in classes):
        return "consistent_indications", (
            "no available modality stores a review priority; the stored non-indications are "
            "consistent, which is still not a confirmation of health"
        )
    if all(c == "indicates" for c in classes):
        return "consistent_indications", (
            "every available modality stores a review priority; the stored indications are "
            "consistent, which is not a confirmed fault"
        )
    return "mixed_indications", (
        "at least one modality stores a review priority while another stores none; the sources "
        "are unrelated and agreement is not asserted"
    )


class TemplateAnalyzeProvider:
    """Deterministic offline cross-modality analysis provider (no network, no ML)."""

    provider = "template"
    model: str | None = None

    def analyze(self, assessment: Assessment) -> AiAnalyzeOutput:
        facts = project_assessment(assessment)
        completeness = facts["evidence_completeness"]
        expected = facts["expected_modalities"]
        available = [m for m in facts["modalities"] if m["availability"] == "available"]
        present = [m["kind"] for m in available]
        missing = [k for k in expected if k not in set(present)]

        summaries: list[AiModalitySummary] = []
        classes: list[str] = []
        lines: list[str] = []
        for mod in facts["modalities"]:
            if mod["availability"] == "available" and isinstance(mod["output"], dict):
                klass, phrase = indication(mod["kind"], mod["output"])
                classes.append(klass)
            else:
                klass, phrase = "clear", "no stored indication (no model output exists for this modality)"
            summaries.append(
                AiModalitySummary(
                    kind=mod["kind"],
                    availability=mod["availability"],
                    reason_code=mod["reason_code"],
                    source_mode=mod["source_mode"],
                    evidence_kinds=[e["kind"] for e in mod["evidence"]],
                    limitations=[_clean(item, 600, quotes=False) for item in mod["limitations"]],
                    indication=phrase,
                )
            )
            observed = mod["observed_at"] or "missing measurement timestamp"
            lines.append(
                f'{mod["kind"].capitalize()}: availability "{mod["availability"]}" '
                f'(reason "{mod["reason_code"]}"), source mode "{mod["source_mode"]}"; '
                f"stored indication: {phrase}; stored evidence kinds: "
                f'{", ".join(e["kind"] for e in mod["evidence"]) or "none recorded"}; '
                f"observed at {observed}."
            )

        # ordered pairwise relations between AVAILABLE modalities (canonical order)
        avail_classes = {m["kind"]: klass for m, klass in zip(available, classes)}
        relations: list[AiPairRelation] = []
        ordered = [k for k in CANONICAL_KINDS if k in avail_classes]
        for i, kind_a in enumerate(ordered):
            for kind_b in ordered[i + 1:]:
                relation = _pair_relation(
                    kind_a, avail_classes[kind_a], kind_b, avail_classes[kind_b]
                )
                relations.append(
                    AiPairRelation(kinds=[kind_a, kind_b], relation=relation)  # type: ignore[list-item]
                )

        agreement, agreement_note = _agreement(classes)

        indicators: list[str] = []
        if missing:
            indicators.append(
                "A requested modality is missing, which limits cross-model interpretation; "
                "missing modalities are recorded, not interpreted."
            )
        for flag in facts["quality_flags"]:
            if flag["code"] == "BEARING_UNCERTAIN":
                indicators.append(
                    "Stored bearing classifier uncertainty limits pairwise interpretation; a "
                    "conservative review status is reported, not a confirmed fault."
                )
            elif flag["code"] == "MIXED_SOURCE_PROVENANCE":
                indicators.append(
                    "Available evidence combines different source modes; the sources are "
                    "unrelated, so combined interpretation carries a stored caution."
                )
            elif flag["code"] == "TIMESTAMP_MISSING":
                indicators.append(
                    "A measurement timestamp is missing; no age or freshness judgement is made."
                )
            elif flag["code"] == "SOURCE_ILLUSTRATIVE":
                indicators.append(
                    "At least one input is illustrative demo/simulated data, not a "
                    "real-machine measurement."
                )
        conservative = [
            "Co-occurrence of stored indications is described, never combined into one score.",
            "watch, unknown and anomalous statuses are conservative review priorities, not "
            "confirmed faults.",
            "No modality explains another and no causality is inferred anywhere in this analysis.",
        ]

        modes = ", ".join(f'{m["kind"]} "{m["source_mode"]}"' for m in facts["modalities"])
        flag_codes = [f["code"] for f in facts["quality_flags"]]
        provenance = (
            f'Assessment-level source mode is "{assessment.source_mode}". Modality source modes: '
            f'{modes}. Mixed provenance flag: '
            f'{"stored" if "MIXED_SOURCE_PROVENANCE" in flag_codes else "not stored"}. '
            f'Illustrative-source flag: {"stored" if "SOURCE_ILLUSTRATIVE" in flag_codes else "not stored"}.'
        )
        completeness_note = (
            f"Requested modalities: {', '.join(expected)}; {completeness['present']} of "
            f"{completeness['expected']} requested modalities produced available evidence; missing "
            f"modalities: {', '.join(missing) if missing else 'none'}. This is the requested "
            "multimodal scope only: it is not machine-health completeness or coverage."
        )
        reasons_part = ", ".join(
            f'"{_clean(r, 200, quotes=False)}"' for r in facts["status_reasons"]
        )
        header = (
            f"Cross-model analysis of assessment {assessment.assessment_id} for machine "
            f'{facts["machine_id"]}. Stored overall status is "{facts["overall_status"]}" '
            f"(stored reasons: {reasons_part or 'none recorded'})."
        )
        relation_lines = [
            f"- {r.kinds[0]} + {r.kinds[1]}: {r.relation}." for r in relations
        ]
        sections = [
            header,
            completeness_note,
            *lines,
            "Pairwise relations between available modalities (stored indications only): "
            + ("; ".join(relation_lines) if relation_lines else "none available."),
            f"Agreement summary: {agreement} - {agreement_note}.",
            "Conservative interpretation: " + " ".join(conservative),
            "Uncertainty indicators: "
            + ("; ".join(indicators) if indicators else "none stored."),
            "Provenance: " + provenance,
            "Non-claims: " + " ".join(NON_CLAIMS),
            FOOTER,
        ]
        analysis_text = _clean(" ".join(sections))[:MAX_EXPLANATION_CHARS]
        evidence_ids = [e["evidence_id"] for m in facts["modalities"] for e in m["evidence"]]
        return AiAnalyzeOutput(
            feature="analyze",
            assessment_id=str(assessment.assessment_id),
            machine_id=assessment.machine_id,
            overall_status=assessment.overall_status,
            status_reasons=list(assessment.status_reasons),
            expected_modalities=list(assessment.expected_modalities),
            present_modalities=present,
            missing_modalities=missing,
            completeness_note=completeness_note,
            modality_summaries=summaries,
            relations=relations,
            agreement=agreement,
            agreement_note=agreement_note,
            conservative_interpretation=conservative,
            uncertainty_indicators=indicators,
            provenance_summary=provenance,
            non_claims=list(NON_CLAIMS),
            analysis_text=analysis_text,
            referenced_assessment_ids=[str(assessment.assessment_id)],
            referenced_evidence_ids=evidence_ids,
        )

    def prompt_sha256(self, assessment: Assessment) -> str:
        """Deterministic digest of the effective prompt (version + instructions + facts)."""
        canonical = json.dumps(
            {
                "prompt_version": PROMPT_VERSION_ANALYZE,
                "instructions": SYSTEM_INSTRUCTIONS,
                "facts": project_assessment(assessment),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _base(assessment: Assessment, text: str, agreement: AgreementValue, note: str) -> AiAnalyzeOutput:
    facts = project_assessment(assessment)
    expected = facts["expected_modalities"]
    present = [m["kind"] for m in facts["modalities"] if m["availability"] == "available"]
    missing = [k for k in expected if k not in set(present)]
    return AiAnalyzeOutput(
        feature="analyze",
        assessment_id=str(assessment.assessment_id),
        machine_id=assessment.machine_id,
        overall_status=assessment.overall_status,
        status_reasons=list(assessment.status_reasons),
        expected_modalities=list(assessment.expected_modalities),
        present_modalities=present,
        missing_modalities=missing,
        completeness_note=(
            "Only the stored scope and status are reported; no cross-model interpretation is "
            "offered beyond them."
        ),
        modality_summaries=[],
        relations=[],
        agreement=agreement,
        agreement_note=note,
        conservative_interpretation=[
            "No cross-model interpretation is offered; co-occurrence is not described here."
        ],
        uncertainty_indicators=["No cross-model interpretation is offered."],
        provenance_summary=(
            f'Assessment-level source mode is "{assessment.source_mode}"; per-modality provenance '
            "was not analysed."
        ),
        non_claims=list(NON_CLAIMS),
        analysis_text=text,
        referenced_assessment_ids=[str(assessment.assessment_id)],
        referenced_evidence_ids=[],
    )


def refusal_output_analyze(assessment: Assessment) -> AiAnalyzeOutput:
    """Deterministic refusal: fewer than two available modalities."""
    return _base(
        assessment,
        f"Assessment {assessment.assessment_id} for machine {assessment.machine_id} has fewer "
        f'than two available modalities, so no cross-model analysis is offered. The stored '
        f'overall status is "{assessment.overall_status}". Use the explain or evidence-review '
        "endpoints for a single-modality snapshot. " + FOOTER,
        "indeterminate",
        "fewer than two available modalities are stored, so cross-model agreement is not assessed",
    )


def fallback_output_analyze(assessment: Assessment) -> AiAnalyzeOutput:
    """Deterministic fallback: generation or validation failed."""
    return _base(
        assessment,
        f"Assessment {assessment.assessment_id} for machine {assessment.machine_id} has stored "
        f'overall status "{assessment.overall_status}". A full cross-model analysis could not be '
        "produced, so only the stored scope and status are reported and no interpretation is "
        "offered. " + FOOTER,
        "indeterminate",
        "the analysis did not validate, so cross-model agreement is not assessed",
    )
