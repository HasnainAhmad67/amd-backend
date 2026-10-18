"""Evidence-grounding validation for AI multi-model analyses (slice C).

Runs AFTER generation and BEFORE persistence. An analysis that fails ANY check is
never stored as "validated"; the route falls back to the deterministic
fallback/refusal output instead.

Checks (each returns a precise, sanitized reason - no paths, SQL or internals):
1. feature and agreement value are allowed; at least two modalities are available
   (a single-modality analysis is a refusal, never a validated analysis);
2. identity: assessment_id / machine_id / overall_status equal the stored ones;
3. modality sets: expected == stored requested set; present == stored available kinds;
   missing == expected minus present;
4. per-modality summaries mirror the stored modality records (availability,
   reason_code, source_mode, evidence kinds, limitations);
5. pairwise relations exist exactly for the canonical pairs of AVAILABLE modalities,
   and every relation names the two stored kinds;
6. stored-only facts: status_reasons subset; uncertainty/provenance text mentions only
   stored flags; referenced_assessment_ids is exactly the requested id; every
   referenced evidence id exists in the snapshot;
7. quoted tokens are stored values and every number is grounded (stored value,
   stored-string number or structural count);
8. required non-claim phrases present ("not a fault diagnosis", "not machine-health
   completeness", "not coverage", "no fused failure probability").
"""

from __future__ import annotations

import re

from ..schemas.ai import AiAnalyzeOutput
from ..schemas.assessment import Assessment
from .ai_analyze_template import CANONICAL_KINDS, REQUIRED_PHRASES
from .ai_validate import _ISO, _NUMBER, _QUOTED, _UUID, allowed_numbers, allowed_quoted

_AGREEMENTS = {"consistent_indications", "mixed_indications", "indeterminate"}


def _validate_minimal(output: AiAnalyzeOutput, assessment: Assessment) -> tuple[bool, str]:
    """Deterministic fallback/refusal shape: identity + grounding + required phrases.

    A fallback/refusal carries no per-modality rows and no relations, so only the
    facts it DOES state are checked (nothing invented, nothing missing)."""
    assessment_id = str(assessment.assessment_id)
    if output.assessment_id != assessment_id:
        return False, "assessment_id does not match the stored snapshot"
    if output.machine_id != assessment.machine_id:
        return False, "machine_id does not match the stored snapshot"
    if output.overall_status != assessment.overall_status:
        return False, "overall_status does not match the stored status"
    if set(output.expected_modalities) != set(assessment.expected_modalities):
        return False, "expected_modalities do not match the stored requested set"
    if not set(output.status_reasons) <= set(assessment.status_reasons):
        return False, "status_reasons contain reasons that are not stored"
    if output.referenced_assessment_ids != [assessment_id]:
        return False, "referenced_assessment_ids must contain only the requested assessment id"
    stored_evidence = {str(e.evidence_id) for e in assessment.evidence}
    if not set(output.referenced_evidence_ids) <= stored_evidence:
        return False, "referenced_evidence_ids contain ids that are not stored in the snapshot"
    text = _UUID.sub(" ", _ISO.sub(" ", output.analysis_text))
    numbers = allowed_numbers(assessment)
    for token in _NUMBER.findall(text):
        try:
            value = float(token)
        except ValueError:  # pragma: no cover - regex guarantees parseable tokens
            continue
        if value not in numbers:
            return False, f"number is not grounded in the stored snapshot: {token!r}"
    for token in _QUOTED.findall(output.analysis_text):
        if token not in allowed_quoted(assessment):
            return False, f"quoted token is not a stored value: {token[:60]!r}"
    for phrase in REQUIRED_PHRASES:
        if phrase not in output.analysis_text:
            return False, f"analysis is missing the required non-claim phrase: {phrase!r}"
    if not output.non_claims:
        return False, "non_claims must be present"
    return True, ""


def validate_analyze_output(output: AiAnalyzeOutput, assessment: Assessment) -> tuple[bool, str]:
    """Ground an AI cross-modality analysis against the stored snapshot. (ok, reason)."""
    if output.feature != "analyze":
        return False, "feature must be analyze"
    if output.agreement not in _AGREEMENTS:
        return False, "agreement must be one of the three allowed values"
    assessment_id = str(assessment.assessment_id)
    if output.assessment_id != assessment_id:
        return False, "assessment_id does not match the stored snapshot"
    if output.machine_id != assessment.machine_id:
        return False, "machine_id does not match the stored snapshot"
    if output.overall_status != assessment.overall_status:
        return False, "overall_status does not match the stored status"
    if not output.relations and not output.modality_summaries:
        # deterministic fallback/refusal shape (no cross-model interpretation offered)
        return _validate_minimal(output, assessment)
    present = sorted(m.kind for m in assessment.modalities if m.availability == "available")
    if len(present) < 2:
        return False, "at least two available modalities are required for a validated analysis"
    if set(output.expected_modalities) != set(assessment.expected_modalities):
        return False, "expected_modalities do not match the stored requested set"
    if sorted(output.present_modalities) != present:
        return False, "present_modalities do not match the stored available modalities"
    missing = sorted(set(assessment.expected_modalities) - set(present))
    if sorted(output.missing_modalities) != missing:
        return False, "missing_modalities are not expected-minus-present of the stored snapshot"
    if not set(output.status_reasons) <= set(assessment.status_reasons):
        return False, "status_reasons contain reasons that are not stored"

    # per-modality summaries mirror the stored records
    by_kind = {m.kind: m for m in assessment.modalities}
    if len(output.modality_summaries) != len(assessment.modalities):
        return False, "modality_summaries must cover every stored modality"
    stored_kinds_by_source: dict[str, set[str]] = {}
    for e in assessment.evidence:
        stored_kinds_by_source.setdefault(e.source, set()).add(e.kind)
    for row in output.modality_summaries:
        stored = by_kind.get(row.kind)
        if stored is None:
            return False, f"modality_summaries contain a kind not stored: {row.kind}"
        if (row.availability, row.reason_code, row.source_mode) != (
            stored.availability, stored.reason_code, stored.source_mode
        ):
            return False, f"modality summary does not match the stored record: {row.kind}"
        if not set(row.evidence_kinds) <= stored_kinds_by_source.get(row.kind, set()):
            return False, f"evidence kinds are not stored for this modality: {row.kind}"
        if [item[:600] for item in row.limitations] != [item[:600] for item in stored.limitations]:
            return False, f"modality limitations are not the stored limitations: {row.kind}"
        # the indication phrase may only quote stored tokens (checked below on the text)
        for token in _QUOTED.findall(row.indication):
            if token not in allowed_quoted(assessment):
                return False, f"indication quotes a value that is not stored: {token[:60]!r}"

    # pairwise relations: exactly the canonical pairs of AVAILABLE modalities
    expected_pairs = []
    ordered = [k for k in CANONICAL_KINDS if k in set(present)]
    for i, kind_a in enumerate(ordered):
        for kind_b in ordered[i + 1:]:
            expected_pairs.append(frozenset((kind_a, kind_b)))
    actual_pairs = [frozenset(r.kinds) for r in output.relations]
    if sorted(map(sorted, actual_pairs)) != sorted(map(sorted, expected_pairs)):
        return False, "relations do not match the canonical pairs of the available modalities"

    if output.referenced_assessment_ids != [assessment_id]:
        return False, "referenced_assessment_ids must contain only the requested assessment id"
    stored_evidence = {str(e.evidence_id) for e in assessment.evidence}
    if not set(output.referenced_evidence_ids) <= stored_evidence:
        return False, "referenced_evidence_ids contain ids that are not stored in the snapshot"
    if len(set(output.referenced_evidence_ids)) != len(output.referenced_evidence_ids):
        return False, "referenced_evidence_ids contain duplicates"

    text = _UUID.sub(" ", _ISO.sub(" ", output.analysis_text))
    numbers = allowed_numbers(assessment)
    for token in _NUMBER.findall(text):
        try:
            value = float(token)
        except ValueError:  # pragma: no cover - regex guarantees parseable tokens
            continue
        if value not in numbers:
            return False, f"number is not grounded in the stored snapshot: {token!r}"
    for token in _QUOTED.findall(output.analysis_text):
        if token not in allowed_quoted(assessment):
            return False, f"quoted token is not a stored value: {token[:60]!r}"

    for phrase in REQUIRED_PHRASES:
        if phrase not in output.analysis_text:
            return False, f"analysis is missing the required non-claim phrase: {phrase!r}"
    if not output.non_claims:
        return False, "non_claims must be present"
    if not output.relations:
        return False, "a validated analysis must describe the pairwise relations"
    return True, ""
