"""Central AI Analysis Agent - slice B: Evidence & Uncertainty Review (deterministic).

Deterministic, offline review provider. It reuses the slice-A facts-only projection
(``ai_template.project_assessment``) so both AI features read the snapshot through the
exact same boundary: no ML calls, no raw audio, no fleet telemetry, no filesystem
paths, and no new score, p-value, confidence, status, id or timestamp.

Uncertainty is EXPLAINED with fixed conservative templates and never turned into a
fault claim: every review ends with the fixed non-claim footer (not a fault
diagnosis, not machine-health completeness, not coverage).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from ..schemas.ai import (
    MAX_EXPLANATION_CHARS,
    PROMPT_VERSION_REVIEW,
    AiEvidenceReviewOutput,
    AiFlagEntry,
    AiModalityReview,
)
from ..schemas.assessment import Assessment
from .ai_template import SYSTEM_INSTRUCTIONS, _clean, project_assessment

# Fixed conservative meanings for STORED statuses and STORED flag codes. Nothing here
# is derived from a model: each sentence only describes what the snapshot already says.
STATUS_MEANING = {
    "normal": 'The stored overall status is "{status}": no model-indicated review priority is recorded.',
    "watch": 'The stored overall status is "{status}": a conservative review priority, '
             "not a confirmed fault.",
    "critical": 'The stored overall status is "{status}": a model-indicated review priority, '
                "not a confirmed fault.",
    "unknown": 'The stored overall status is "{status}": evidence is incomplete, ambiguous or missing, so no '
               "normal interpretation is offered.",
}

# Fixed conservative meaning for STORED modality-level statuses (e.g. the acoustic
# model's own status vocabulary), never a fault claim.
MODALITY_STATUS_MEANING = {
    "watch": 'a stored "{status}" model status is a conservative review priority, not a confirmed fault',
    "anomalous": 'a stored "{status}" model status is a conservative anomaly indication, not a '
                 "confirmed fault",
}

FLAG_MEANING = {
    "EVIDENCE_INCOMPLETE": "Fewer modalities than requested produced available evidence; completeness "
                           "measures presence, not confidence.",
    "NO_AVAILABLE_MODALITY": "No modality produced available evidence; the snapshot cannot be interpreted.",
    "SOURCE_ILLUSTRATIVE": "At least one input is dataset_demo/simulated illustrative demo data, not a "
                           "real-machine measurement.",
    "MIXED_SOURCE_PROVENANCE": "Available evidence combines different source modes; the sources are "
                               "unrelated, so interpret the combined status with care.",
    "TIMESTAMP_MISSING": "A measurement timestamp is missing; no age or freshness judgement is made and none "
                         "was fabricated.",
    "MODEL_IDENTITY_INCOMPLETE": "Model identity is incomplete (missing artifact or preprocessing identity); "
                                 "the missing metadata was not invented.",
    "BEARING_UNCERTAIN": "The bearing classifier output is ambiguous, empty or inconsistent; a conservative "
                         "review status is reported, not a confirmed fault.",
    "MODEL_LIMITATIONS_PRESENT": "Model applicability and generalization limitations are attached to this "
                                 "evidence; no accuracy guarantee is implied.",
    "AUDIO_TRUNCATED": "The model processing cap truncated scoring for an upload; only part of it "
                       "was scored.",
}

# Fixed non-claims, identical on every review (structured field + text footer).
NON_CLAIMS = [
    "This review is not a fault diagnosis and does not confirm any fault.",
    "This review is not machine-health completeness and makes no coverage claim.",
    "Missing or uncertain evidence limits interpretation; nothing was fabricated to fill gaps.",
    "No score, p-value or confidence is created here; stored model values are quoted only.",
]

FOOTER = (
    "This review reports stored evidence and uncertainty only. It is not a fault diagnosis, "
    "not machine-health completeness, and not coverage."
)

REQUIRED_PHRASES = ("not a fault diagnosis", "not machine-health completeness", "not coverage")


def _missing(bits: list[str], present: list[str]) -> list[str]:
    """Missing modalities = expected minus present (stored lists only)."""
    have = set(present)
    return [k for k in bits if k not in have]


class TemplateEvidenceReviewProvider:
    """Deterministic offline review provider: no network, no randomness, no ML."""

    provider = "template"
    model: str | None = None

    def review(self, assessment: Assessment) -> AiEvidenceReviewOutput:
        facts = project_assessment(assessment)
        completeness = facts["evidence_completeness"]
        expected = facts["expected_modalities"]
        present = [m["kind"] for m in facts["modalities"] if m["availability"] == "available"]
        missing = _missing(expected, present)
        stored_flags = facts["quality_flags"]

        modality_reviews: list[AiModalityReview] = []
        lines: list[str] = []
        for mod in facts["modalities"]:
            modality_reviews.append(
                AiModalityReview(
                    kind=mod["kind"],
                    availability=mod["availability"],
                    reason_code=mod["reason_code"],
                    source_mode=mod["source_mode"],
                    observed_at=mod["observed_at"],
                    inference_at=mod["inference_at"],
                    model_name=mod["model_name"],
                    model_version=mod["model_version"],
                    training_dataset=mod["training_dataset"],
                    evidence_kinds=[e["kind"] for e in mod["evidence"]],
                    limitations=[_clean(item, 600, quotes=False) for item in mod["limitations"]],
                )
            )
            observed = mod["observed_at"] or "missing measurement timestamp"
            inferred = mod["inference_at"] or "no recorded inference time"
            if mod["availability"] == "available":
                identity = (
                    f'model {mod["model_name"] or "unnamed model"} '
                    f'version {mod["model_version"] or "no version"}, trained on '
                    f'{mod["training_dataset"] or "no training dataset recorded"}'
                )
                kinds = ", ".join(e["kind"] for e in mod["evidence"]) or "none recorded"
                limits = "; ".join(f"- {_clean(item, 600, quotes=False)}" for item in mod["limitations"])
                lines.append(
                    f'{mod["kind"].capitalize()}: availability "{mod["availability"]}" '
                    f'(reason "{mod["reason_code"]}"), source mode "{mod["source_mode"]}"; '
                    f"observed at {observed}; inferred at {inferred}; {identity}; "
                    f"stored evidence kinds: {kinds}; stored limitations: {limits or 'none stored'}."
                )
            else:
                lines.append(
                    f'{mod["kind"].capitalize()}: availability "{mod["availability"]}" '
                    f'(reason "{mod["reason_code"]}"), source mode "{mod["source_mode"]}"; '
                    "no model output exists for this modality, so nothing is interpreted from it."
                )

        indicators: list[str] = []
        indicators.append(STATUS_MEANING[facts["overall_status"]].format(status=facts["overall_status"]))
        for mod in facts["modalities"]:
            output = mod.get("output") if isinstance(mod.get("output"), dict) else {}
            modal_status = output.get("status") or output.get("risk_level")
            if modal_status in MODALITY_STATUS_MEANING:
                indicators.append(
                    f'The {mod["kind"]} model: '
                    + MODALITY_STATUS_MEANING[modal_status].format(status=modal_status)
                    + "."
                )
        for flag in stored_flags:
            meaning = FLAG_MEANING.get(flag["code"])
            if meaning:
                indicators.append(f'"{flag["code"]}" ({flag["severity"]}): {meaning}')
        if missing:
            indicators.append(
                "Not every requested modality produced available evidence; missing modalities are "
                "recorded, not interpreted."
            )

        modes = ", ".join(f'{m["kind"]} "{m["source_mode"]}"' for m in facts["modalities"])
        flag_codes = [f["code"] for f in stored_flags]
        provenance = (
            f'Assessment-level source mode is "{facts_source_mode(assessment)}". '
            f"Modality source modes: {modes}. Mixed provenance flag: "
            f'{"stored" if "MIXED_SOURCE_PROVENANCE" in flag_codes else "not stored"}. '
            f'Illustrative-source flag: {"stored" if "SOURCE_ILLUSTRATIVE" in flag_codes else "not stored"}.'
        )

        completeness_note = (
            f"Requested modalities: {', '.join(expected)}; {completeness['present']} of "
            f"{completeness['expected']} requested modalities produced available evidence; missing "
            f"modalities: {', '.join(missing) if missing else 'none'}. This is the requested multimodal "
            "scope only: it is not machine-health completeness or coverage."
        )
        reasons_part = ", ".join(f'"{_clean(r, 200, quotes=False)}"' for r in facts["status_reasons"])
        header = (
            f"Evidence and uncertainty review of assessment {assessment.assessment_id} for machine "
            f'{facts["machine_id"]}. Stored overall status is "{facts["overall_status"]}" '
            f"(stored reasons: {reasons_part or 'none recorded'})."
        )
        flag_lines = [
            f'"{f["code"]}" ({f["severity"]}) - {_clean(f["explanation"], 400, quotes=False)}'
            for f in stored_flags
        ]
        sections = [
            header,
            completeness_note,
            *lines,
            "Stored quality flags: " + ("; ".join(flag_lines) if flag_lines else "none stored."),
            "Uncertainty indicators: " + ("; ".join(indicators) if indicators else "none stored."),
            "Provenance: " + provenance,
            "Non-claims: " + " ".join(NON_CLAIMS),
            FOOTER,
        ]
        review_text = _clean(" ".join(sections))[:MAX_EXPLANATION_CHARS]

        evidence_ids = [e["evidence_id"] for m in facts["modalities"] for e in m["evidence"]]
        return AiEvidenceReviewOutput(
            feature="evidence_review",
            assessment_id=str(assessment.assessment_id),
            machine_id=assessment.machine_id,
            overall_status=assessment.overall_status,
            status_reasons=list(assessment.status_reasons),
            expected_modalities=list(assessment.expected_modalities),
            present_modalities=present,
            missing_modalities=missing,
            completeness_note=completeness_note,
            modality_reviews=modality_reviews,
            quality_flags=[AiFlagEntry(code=f["code"], severity=f["severity"]) for f in stored_flags],
            uncertainty_indicators=indicators,
            provenance_summary=provenance,
            non_claims=list(NON_CLAIMS),
            review_text=review_text,
            referenced_assessment_ids=[str(assessment.assessment_id)],
            referenced_evidence_ids=evidence_ids,
        )

    def prompt_sha256(self, assessment: Assessment) -> str:
        """Deterministic digest of the effective prompt (version + instructions + facts)."""
        canonical = json.dumps(
            {
                "prompt_version": PROMPT_VERSION_REVIEW,
                "instructions": SYSTEM_INSTRUCTIONS,
                "facts": project_assessment(assessment),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def facts_source_mode(assessment: Assessment) -> str:
    """Assessment-level source mode, straight from the stored snapshot."""
    return assessment.source_mode


def refusal_output_review(assessment: Assessment) -> AiEvidenceReviewOutput:
    """Deterministic refusal: not enough available evidence to review anything."""
    facts = project_assessment(assessment)
    expected = facts["expected_modalities"]
    present = [m["kind"] for m in facts["modalities"] if m["availability"] == "available"]
    missing = _missing(expected, present)
    return AiEvidenceReviewOutput(
        feature="evidence_review",
        assessment_id=str(assessment.assessment_id),
        machine_id=assessment.machine_id,
        overall_status=assessment.overall_status,
        status_reasons=list(assessment.status_reasons),
        expected_modalities=list(assessment.expected_modalities),
        present_modalities=present,
        missing_modalities=missing,
        completeness_note=(
            "No modality produced available evidence, so this snapshot cannot be reviewed; the "
            "requested scope is recorded but no interpretation is offered."
        ),
        modality_reviews=[],
        quality_flags=[AiFlagEntry(code=f["code"], severity=f["severity"]) for f in facts["quality_flags"]],
        uncertainty_indicators=["No available evidence was stored in this snapshot."],
        provenance_summary=(
            f'Assessment-level source mode is "{assessment.source_mode}"; no modality evidence '
            "is available to summarize."
        ),
        non_claims=list(NON_CLAIMS),
        review_text=(
            f"Assessment {assessment.assessment_id} for machine {assessment.machine_id} does not contain "
            f'enough available evidence to review. The stored overall status is '
            f'"{assessment.overall_status}" and no further interpretation is offered, because missing '
            "evidence limits interpretation. " + FOOTER
        ),
        referenced_assessment_ids=[str(assessment.assessment_id)],
        referenced_evidence_ids=[],
    )


def fallback_output_review(assessment: Assessment) -> AiEvidenceReviewOutput:
    """Deterministic fallback: generation or validation failed; report stored facts only."""
    facts = project_assessment(assessment)
    expected = facts["expected_modalities"]
    present = [m["kind"] for m in facts["modalities"] if m["availability"] == "available"]
    missing = _missing(expected, present)
    return AiEvidenceReviewOutput(
        feature="evidence_review",
        assessment_id=str(assessment.assessment_id),
        machine_id=assessment.machine_id,
        overall_status=assessment.overall_status,
        status_reasons=list(assessment.status_reasons),
        expected_modalities=list(assessment.expected_modalities),
        present_modalities=present,
        missing_modalities=missing,
        completeness_note=(
            "A full review could not be produced for this assessment; only the stored scope and "
            "status are reported, with no interpretation beyond them."
        ),
        modality_reviews=[],
        quality_flags=[AiFlagEntry(code=f["code"], severity=f["severity"]) for f in facts["quality_flags"]],
        uncertainty_indicators=["The review template did not validate; no interpretation is offered."],
        provenance_summary=(
            f'Assessment-level source mode is "{assessment.source_mode}"; per-modality provenance was '
            "not reviewed because the review did not validate."
        ),
        non_claims=list(NON_CLAIMS),
        review_text=(
            f"Assessment {assessment.assessment_id} for machine {assessment.machine_id} has stored overall "
            f'status "{assessment.overall_status}". A full review could not be produced, so only the stored '
            "scope and status are reported and no interpretation is offered. " + FOOTER
        ),
        referenced_assessment_ids=[str(assessment.assessment_id)],
        referenced_evidence_ids=[],
    )
