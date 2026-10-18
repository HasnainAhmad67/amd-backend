"""Evidence-grounding validation for AI maintenance reports (slice D).

Runs AFTER generation and BEFORE persistence. A report that fails ANY check is never
stored as "validated"; the route falls back to the deterministic fallback/refusal output
instead.

Checks (each returns a precise, sanitized reason - no paths, SQL or internals):
1. identity: machine_id and the derived window (start/end) must match the retrieved
   snapshots and the resolved lookback window;
2. selection: assessment_count must equal the retrieved window-filtered count and
   referenced_assessment_ids must be exactly the retrieved snapshots (owner-visible,
   requested machine, inside the window);
3. references: every referenced evidence id exists in the retrieved snapshots (no dups);
4. full reports: status distribution, per-modality availability and recurring flags must
   equal the values recomputed from the retrieved snapshots;
5. quoted tokens are stored values and every number is grounded (stored value, number
   inside a stored string, or a structural count of the retrieved history);
6. required non-claim phrases present; forbidden claim wording absent.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..schemas.ai import AiReportOutput
from ..schemas.assessment import Assessment
from .ai_report_template import (
    FORBIDDEN_PHRASES,
    REQUIRED_PHRASES,
    modality_availability,
    recurring_flags,
    status_distribution,
)
from .ai_validate import (
    _ISO,
    _NUMBER,
    _QUOTED,
    _UUID,
    _numbers_in_strings,
    allowed_numbers,
    allowed_quoted,
)


def _norm(dt: datetime | None) -> datetime | None:
    return dt.astimezone(timezone.utc) if dt is not None else None


def _allowed_numbers(
    snapshots: list[Assessment], *, machine_id: str, lookback_days: int
) -> set[float]:
    """Stored values/strings + every structural count the report may print."""
    numbers: set[float] = set()
    for snapshot in snapshots:
        numbers |= allowed_numbers(snapshot)
    numbers.update(float(row.count) for row in status_distribution(snapshots))
    for row in modality_availability(snapshots):
        numbers.update(float(count.count) for count in row.counts)
    for flag in recurring_flags(snapshots):
        numbers.add(float(flag.count))
    numbers.add(float(len(snapshots)))
    numbers.add(float(lookback_days))
    numbers |= _numbers_in_strings([machine_id])
    return numbers


def _allowed_quoted(snapshots: list[Assessment]) -> set[str]:
    tokens: set[str] = set()
    for snapshot in snapshots:
        tokens |= allowed_quoted(snapshot)
    return tokens


def _prose_strings(output: AiReportOutput) -> list[str]:
    strings = [output.report_text]
    strings += list(output.missing_data_patterns)
    strings += list(output.conservative_observations)
    strings += list(output.limitations)
    strings += list(output.uncertainty_notes)
    strings += list(output.non_claims)
    for suggestion in output.review_suggestions:
        strings += [suggestion.title, suggestion.basis, suggestion.detail]
    return strings


def _ground_text(
    output: AiReportOutput, snapshots: list[Assessment], *, machine_id: str, lookback_days: int
) -> tuple[bool, str]:
    numbers = _allowed_numbers(snapshots, machine_id=machine_id, lookback_days=lookback_days)
    tokens = _allowed_quoted(snapshots)
    for text in _prose_strings(output):
        stripped = _UUID.sub(" ", _ISO.sub(" ", text))
        for token in _NUMBER.findall(stripped):
            try:
                value = float(token)
            except ValueError:  # pragma: no cover - regex guarantees parseable tokens
                continue
            if value not in numbers:
                return False, f"number is not grounded in the retrieved history: {token!r}"
        for token in _QUOTED.findall(text):
            if token not in tokens:
                return False, f"quoted token is not a stored value: {token[:60]!r}"
    for phrase in FORBIDDEN_PHRASES:
        if phrase in output.report_text:
            return False, f"report contains forbidden claim wording: {phrase!r}"
    for phrase in REQUIRED_PHRASES:
        if phrase not in output.report_text:
            return False, f"report is missing the required non-claim phrase: {phrase!r}"
    if not output.non_claims:
        return False, "non_claims must be present"
    return True, ""


def validate_report_output(
    output: AiReportOutput,
    snapshots: list[Assessment],
    *,
    machine_id: str,
    window_start: datetime | None,
    window_end: datetime | None,
    history_truncated: bool,
    lookback_days: int,
) -> tuple[bool, str]:
    """Ground an AI maintenance report against the retrieved snapshots. (ok, reason)."""
    if output.feature != "report":
        return False, "feature must be report"
    if output.machine_id != machine_id:
        return False, "machine_id does not match the requested machine"
    if _norm(output.window_start) != _norm(window_start) or _norm(output.window_end) != _norm(window_end):
        return False, "window does not match the stored created_at range"
    if output.history_truncated != history_truncated:
        return False, "history_truncated does not match the resolved retrieval"
    ids = [str(snapshot.assessment_id) for snapshot in snapshots]
    if sorted(output.referenced_assessment_ids) != sorted(ids) or len(output.referenced_assessment_ids) != len(ids):
        return False, "referenced_assessment_ids must be exactly the retrieved assessments"
    if output.assessment_count != len(snapshots):
        return False, "assessment_count does not match the retrieved window-filtered count"

    stored_evidence = {str(item.evidence_id) for snapshot in snapshots for item in snapshot.evidence}
    if not set(output.referenced_evidence_ids) <= stored_evidence:
        return False, "referenced_evidence_ids contain ids that are not stored in the retrieval"
    if len(set(output.referenced_evidence_ids)) != len(output.referenced_evidence_ids):
        return False, "referenced_evidence_ids contain duplicates"

    is_full = bool(
        output.modality_availability or output.status_distribution or output.recurring_flags
    )
    if is_full:
        expected_status = [(row.status, row.count) for row in status_distribution(snapshots)]
        actual_status = [(row.status, row.count) for row in output.status_distribution]
        if actual_status != expected_status:
            return False, "status distribution does not match the retrieved snapshots"
        if output.modality_availability != modality_availability(snapshots):
            return False, "modality availability summary does not match the retrieved snapshots"
        if output.recurring_flags != recurring_flags(snapshots):
            return False, "recurring flags do not match the stored quality flags"

    return _ground_text(
        output, snapshots, machine_id=machine_id, lookback_days=lookback_days
    )
