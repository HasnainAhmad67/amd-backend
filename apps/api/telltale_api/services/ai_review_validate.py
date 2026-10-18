"""Evidence-grounding validation for AI evidence reviews (slice B).

Runs AFTER generation and BEFORE persistence. A review that fails ANY check is never
stored as "validated"; the route falls back to the deterministic fallback/refusal
output instead.

Checks (each returns a precise, sanitized reason - no paths, SQL or internals):
1. identity: assessment_id / machine_id / overall_status must equal the stored ones;
2. modality sets: expected == stored requested set; present == stored available kinds;
   missing == expected minus present (nothing invented, nothing hidden);
3. per-modality rows: availability/reason_code/source_mode must match the stored
   modality record, evidence_kinds must be stored evidence kinds of that modality,
   and model identity fields must match the stored identity;
4. stored-only facts: status_reasons subset, quality flag codes+severities must match
   the stored flags exactly, uncertainty indicators only describe stored flags;
5. references: referenced_assessment_ids may contain ONLY the requested (owner-visible)
   assessment id, and every referenced evidence id must exist in that snapshot;
6. quoted tokens: every double-quoted token in the review text must be a value that is
   actually stored in the snapshot - no invented classifications;
7. numeric grounding: every number in the review text must be a STORED numeric value, a
   number that occurs inside a stored string, or a structural count of the snapshot;
8. required non-claim phrases present ("not a fault diagnosis", "not machine-health
   completeness", "not coverage").
"""

from __future__ import annotations

import re

from ..schemas.ai import AiEvidenceReviewOutput
from ..schemas.assessment import Assessment
from .ai_review_template import REQUIRED_PHRASES
from .ai_validate import _ISO, _NUMBER, _QUOTED, _UUID, allowed_numbers, allowed_quoted


def validate_review_output(output: AiEvidenceReviewOutput, assessment: Assessment) -> tuple[bool, str]:
    """Ground an AI evidence review against the stored snapshot. (ok, reason)."""
    if output.feature != "evidence_review":
        return False, "feature must be evidence_review"
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
    missing = sorted(set(assessment.expected_modalities) - set(present))
    if sorted(output.missing_modalities) != missing:
        return False, "missing_modalities are not expected-minus-present of the stored snapshot"
    if not set(output.status_reasons) <= set(assessment.status_reasons):
        return False, "status_reasons contain reasons that are not stored"
    stored_flags = {f.code: f.severity for f in assessment.quality_flags}
    review_flags = {f.code: f.severity for f in output.quality_flags}
    if review_flags != stored_flags:
        return False, "quality flags do not match the stored flags (codes and severities)"

    # per-modality rows must mirror the stored modality records exactly
    by_kind = {m.kind: m for m in assessment.modalities}
    if len(output.modality_reviews) != len(assessment.modalities):
        return False, "modality_reviews must cover every stored modality"
    stored_kinds_by_source: dict[str, set[str]] = {}
    for e in assessment.evidence:
        stored_kinds_by_source.setdefault(e.source, set()).add(e.kind)
    for row in output.modality_reviews:
        stored = by_kind.get(row.kind)
        if stored is None:
            return False, f"modality_reviews contain a kind not stored: {row.kind}"
        if (row.availability, row.reason_code, row.source_mode) != (
            stored.availability,
            stored.reason_code,
            stored.source_mode,
        ):
            return False, f"modality row does not match the stored record: {row.kind}"
        if row.observed_at != stored.observed_at or row.inference_at != stored.inference_at:
            return False, f"modality timestamps do not match the stored record: {row.kind}"
        identity = stored.model_identity
        if (row.model_name, row.model_version, row.training_dataset) != (
            identity.name if identity else None,
            identity.version if identity else None,
            identity.training_dataset if identity else None,
        ):
            return False, f"modality identity fields do not match the stored record: {row.kind}"
        if not set(row.evidence_kinds) <= stored_kinds_by_source.get(row.kind, set()):
            return False, f"evidence kinds are not stored for this modality: {row.kind}"
        if [item[:600] for item in row.limitations] != [item[:600] for item in stored.limitations]:
            return False, f"modality limitations are not the stored limitations: {row.kind}"

    if output.referenced_assessment_ids != [assessment_id]:
        return False, "referenced_assessment_ids must contain only the requested assessment id"
    stored_evidence = {str(e.evidence_id) for e in assessment.evidence}
    if not set(output.referenced_evidence_ids) <= stored_evidence:
        return False, "referenced_evidence_ids contain ids that are not stored in the snapshot"
    if len(set(output.referenced_evidence_ids)) != len(output.referenced_evidence_ids):
        return False, "referenced_evidence_ids contain duplicates"

    tokens = allowed_quoted(assessment)
    for token in _QUOTED.findall(output.review_text):
        if token not in tokens:
            return False, f"quoted token is not a stored value: {token[:60]!r}"

    text = _UUID.sub(" ", _ISO.sub(" ", output.review_text))
    numbers = allowed_numbers(assessment)
    for token in _NUMBER.findall(text):
        try:
            value = float(token)
        except ValueError:  # pragma: no cover - regex guarantees parseable tokens
            continue
        if value not in numbers:
            return False, f"number is not grounded in the stored snapshot: {token!r}"

    for phrase in REQUIRED_PHRASES:
        if phrase not in output.review_text:
            return False, f"review is missing the required non-claim phrase: {phrase!r}"
    if "requested" not in output.review_text:
        return False, "review must describe the requested-scope completeness"
    if not output.non_claims:
        return False, "non_claims must be present"
    return True, ""
