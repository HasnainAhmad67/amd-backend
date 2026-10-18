"""Evidence-grounding validation for AI outputs (slice A: explanations).

Runs AFTER generation and BEFORE persistence. An output that fails ANY check is
never stored as "validated"; the route falls back to the deterministic
fallback/refusal output instead.

Checks (each returns a precise, sanitized reason - no paths, SQL or internals):
1. identity: assessment_id / machine_id / overall_status must equal the stored ones;
2. requested set: expected_modalities must equal the stored expected set exactly;
3. present set: present_modalities must equal the stored available kinds;
4. reasons: status_reasons and quality_flag_codes must be subsets of the stored ones;
5. references: referenced_assessment_ids may contain ONLY the requested (owner-visible)
   assessment id, and every referenced evidence id must exist in that snapshot;
6. quoted tokens: every double-quoted token in the explanation must be a value that is
   actually stored in the snapshot (statuses, reason codes, source modes, flag codes,
   stored status strings) - no invented classifications;
7. numeric grounding: every number in the explanation must be a STORED numeric value, a
   number that occurs inside a stored string (verbatim caveats), or a structural count
   of the snapshot itself - invented scores/p-values/confidences are rejected.
"""

from __future__ import annotations

import re
from typing import Any

from ..schemas.ai import AiExplainOutput
from ..schemas.assessment import Assessment

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?")
_QUOTED = re.compile(r'"([^"]{1,120})"')

# output fields whose STRING values are stored status/classification tokens
_STATUS_FIELDS = ("risk_level", "status", "predicted_class", "calibration", "code")


def _walk_numbers(node: Any) -> list[float]:
    if isinstance(node, bool):
        return []
    if isinstance(node, (int, float)):
        return [float(node)]
    if isinstance(node, dict):
        return [n for value in node.values() for n in _walk_numbers(value)]
    if isinstance(node, list):
        return [n for value in node for n in _walk_numbers(value)]
    return []


def _walk_strings(node: Any) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [s for value in node.values() for s in _walk_strings(value)]
    if isinstance(node, list):
        return [s for value in node for s in _walk_strings(value)]
    return []


def _numbers_in_strings(texts: list[str]) -> set[float]:
    found: set[float] = set()
    for text in texts:
        for token in _NUMBER.findall(_UUID.sub(" ", _ISO.sub(" ", text))):
            try:
                found.add(float(token))
            except ValueError:  # pragma: no cover - regex guarantees parseable tokens
                pass
    return found


def allowed_quoted(assessment: Assessment) -> set[str]:
    """Every short token the stored snapshot legitimately contains."""
    tokens: set[str] = {assessment.overall_status}
    tokens.update(assessment.status_reasons)
    tokens.update(f.code for f in assessment.quality_flags)
    # assessment-level provenance is a stored value too (quoted by review summaries)
    tokens.add(assessment.source_mode)
    for m in assessment.modalities:
        tokens.update({m.availability, m.reason_code, m.source_mode})
        if m.output is None:
            continue
        output = m.output.model_dump(mode="json")
        for field in _STATUS_FIELDS:
            value = output.get(field)
            if isinstance(value, str):
                tokens.add(value)
        for mode in output.get("failure_modes") or []:
            if isinstance(mode, dict) and isinstance(mode.get("code"), str):
                tokens.add(mode["code"])
    return tokens


def allowed_numbers(assessment: Assessment) -> set[float]:
    """Stored numeric values + numbers inside stored strings + structural counts."""
    dump = assessment.model_dump(mode="json")
    numbers = set(_walk_numbers(dump))
    numbers |= _numbers_in_strings(_walk_strings(dump))
    present = sum(1 for m in assessment.modalities if m.availability == "available")
    numbers.update(
        {
            float(present),
            float(len(assessment.expected_modalities)),
            float(len(assessment.quality_flags)),
            float(len(assessment.evidence)),
            float(len(assessment.modalities)),
        }
    )
    return numbers


def validate_explain_output(output: AiExplainOutput, assessment: Assessment) -> tuple[bool, str]:
    """Ground an AI explanation against the stored snapshot. (ok, reason)."""
    assessment_id = str(assessment.assessment_id)
    if output.assessment_id != assessment_id:
        return False, "assessment_id does not match the stored snapshot"
    if output.machine_id != assessment.machine_id:
        return False, "machine_id does not match the stored snapshot"
    if output.overall_status != assessment.overall_status:
        return False, "overall_status does not match the stored status"
    if set(output.expected_modalities) != set(assessment.expected_modalities):
        return False, "expected_modalities do not match the stored requested set"
    present = sorted(m.kind for m in assessment.modalities if m.availability == "available")
    if sorted(output.present_modalities) != present:
        return False, "present_modalities do not match the stored available modalities"
    if not set(output.status_reasons) <= set(assessment.status_reasons):
        return False, "status_reasons contain reasons that are not stored"
    if not set(output.quality_flag_codes) <= {f.code for f in assessment.quality_flags}:
        return False, "quality_flag_codes contain codes that are not stored"
    if output.referenced_assessment_ids != [assessment_id]:
        return False, "referenced_assessment_ids must contain only the requested assessment id"
    stored_evidence = {str(e.evidence_id) for e in assessment.evidence}
    if not set(output.referenced_evidence_ids) <= stored_evidence:
        return False, "referenced_evidence_ids contain ids that are not stored in the snapshot"
    if len(set(output.referenced_evidence_ids)) != len(output.referenced_evidence_ids):
        return False, "referenced_evidence_ids contain duplicates"

    allowed_tokens = allowed_quoted(assessment)
    for token in _QUOTED.findall(output.explanation_text):
        if token not in allowed_tokens:
            return False, f"quoted token is not a stored value: {token[:60]!r}"

    text = _UUID.sub(" ", _ISO.sub(" ", output.explanation_text))
    numbers = allowed_numbers(assessment)
    for token in _NUMBER.findall(text):
        try:
            value = float(token)
        except ValueError:  # pragma: no cover - regex guarantees parseable tokens
            continue
        if value not in numbers:
            return False, f"number is not grounded in the stored snapshot: {token!r}"

    if "confirmed fault" not in output.explanation_text or "model indication" not in output.explanation_text:
        return False, "explanation must state that results are model indications, not confirmed faults"
    if "missing" not in output.explanation_text.lower():
        return False, "explanation must acknowledge that missing evidence limits interpretation"
    return True, ""
