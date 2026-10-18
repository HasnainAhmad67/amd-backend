"""Evidence-grounding validation for Ask Telltale answers (slice E).

Runs AFTER generation and BEFORE persistence. An answer that fails ANY check is never
stored as "validated"; the route falls back to the deterministic fallback/refusal
output instead.

Checks (each returns a precise, sanitized reason - no paths, SQL or internals):
1. feature/intent: intent is a supported dictionary intent (or "unsupported"), and
   matched_terms are canonical dictionary terms consistent with the intent;
2. question digest/echo are exactly the expected, bounded values (the echoed question
   is caller-supplied inert data and is EXCLUDED from evidence-grounding scanning);
3. answered/refusal consistency: a validated answer is answered and never refused;
   every refusal carries a reason;
4. scope identity: assessment scope references exactly the requested assessment;
   machine scope references exactly the retrieved window set, window and truncation;
5. assessment_count equals the retrieved context; every referenced assessment/evidence
   id exists in the context (no duplicates);
6. quoted tokens are stored values and every number is grounded (stored value, number
   inside a stored string, or a structural count of the retrieved context);
7. required non-claim phrases present; forbidden claim wording absent.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..schemas.ai import AiAskOutput
from ..schemas.assessment import Assessment
from .ai_ask_template import (
    ALL_KEYWORDS,
    FORBIDDEN_PHRASES,
    INTENT_KEYWORDS,
    INTENT_ORDER,
    REQUIRED_PHRASES,
)
from .ai_report_template import modality_availability, recurring_flags, status_distribution
from .ai_validate import (
    _ISO,
    _NUMBER,
    _QUOTED,
    _UUID,
    _numbers_in_strings,
    allowed_numbers,
    allowed_quoted,
)

_ALLOWED_INTENTS = frozenset(INTENT_ORDER) | {"unsupported"}
_REFUSAL_REASONS = frozenset(
    {"unsupported_question", "evidence_insufficient", "no_evidence", "scope_mismatch"}
)


def _norm(dt: datetime | None) -> datetime | None:
    return dt.astimezone(timezone.utc) if dt is not None else None


def _allowed_numbers(
    context: list[Assessment], *, machine_id: str | None, lookback_days: int | None
) -> set[float]:
    numbers: set[float] = set()
    for snapshot in context:
        numbers |= allowed_numbers(snapshot)
    numbers.update(float(row.count) for row in status_distribution(context))
    for row in modality_availability(context):
        numbers.update(float(count.count) for count in row.counts)
    for flag in recurring_flags(context):
        numbers.add(float(flag.count))
    numbers.add(float(len(context)))
    if lookback_days is not None:
        numbers.add(float(lookback_days))
    if machine_id:
        numbers |= _numbers_in_strings([machine_id])
    return numbers


def _allowed_quoted(context: list[Assessment]) -> set[str]:
    tokens: set[str] = set()
    for snapshot in context:
        tokens |= allowed_quoted(snapshot)
    return tokens


def _prose_strings(output: AiAskOutput) -> list[str]:
    """Prose fields that carry evidence claims. ``question_echo`` and
    ``question_sha256`` are deliberately EXCLUDED (caller-supplied / digest only)."""
    strings = [output.answer_text, output.answer_summary]
    strings += list(output.relevant_facts)
    strings += list(output.uncertainty_indicators)
    strings += list(output.limitations)
    strings += list(output.non_claims)
    return strings


def _ground_text(
    output: AiAskOutput,
    context: list[Assessment],
    *,
    machine_id: str | None,
    lookback_days: int | None,
) -> tuple[bool, str]:
    numbers = _allowed_numbers(context, machine_id=machine_id, lookback_days=lookback_days)
    tokens = _allowed_quoted(context)
    for text in _prose_strings(output):
        stripped = _UUID.sub(" ", _ISO.sub(" ", text))
        for token in _NUMBER.findall(stripped):
            try:
                value = float(token)
            except ValueError:  # pragma: no cover - regex guarantees parseable tokens
                continue
            if value not in numbers:
                return False, f"number is not grounded in the retrieved context: {token!r}"
        for token in _QUOTED.findall(text):
            if token not in tokens:
                return False, f"quoted token is not a stored value: {token[:60]!r}"
    for phrase in FORBIDDEN_PHRASES:
        if phrase in output.answer_text:
            return False, f"answer contains forbidden claim wording: {phrase!r}"
    for phrase in REQUIRED_PHRASES:
        if phrase not in output.answer_text:
            return False, f"answer is missing the required non-claim phrase: {phrase!r}"
    if not output.non_claims:
        return False, "non_claims must be present"
    return True, ""


def validate_ask_output(
    output: AiAskOutput,
    context: list[Assessment],
    *,
    scope: str,
    machine_id: str | None,
    lookback_days: int | None,
    window_start: datetime | None,
    window_end: datetime | None,
    history_truncated: bool,
    question_sha256: str,
    question_echo: str,
) -> tuple[bool, str]:
    """Ground an Ask Telltale answer against the retrieved owner-visible context."""
    if output.feature != "ask":
        return False, "feature must be ask"
    if output.scope != scope:
        return False, "scope does not match the request"
    if output.question_sha256 != question_sha256:
        return False, "question digest does not match the request"
    if output.question_echo != question_echo:
        return False, "question echo does not match the sanitized request"

    # intent + matched_terms must be consistent with the fixed dictionary
    if output.intent not in _ALLOWED_INTENTS:
        return False, "intent is not a supported dictionary intent"
    if output.intent == "unsupported":
        if output.matched_terms:
            return False, "an unsupported intent must not carry matched terms"
    else:
        if not output.matched_terms:
            return False, "a supported intent must carry the matched canonical terms"
        if not set(output.matched_terms) <= set(INTENT_KEYWORDS[output.intent]):
            return False, "matched terms are not canonical for the intent"
        if not set(output.matched_terms) <= ALL_KEYWORDS:
            return False, "matched terms are not in the canonical dictionary"

    # answered / refusal consistency
    if output.answered:
        if output.intent == "unsupported":
            return False, "an unsupported intent cannot be answered"
        if output.refusal_reason is not None:
            return False, "an answered response must not carry a refusal reason"
        if output.intent == "history" and scope != "machine":
            return False, "the history intent is valid only for machine scope"
    else:
        if output.refusal_reason not in _REFUSAL_REASONS:
            return False, "a refusal must carry a supported refusal reason"
        if output.refusal_reason == "unsupported_question" and output.intent != "unsupported":
            return False, "unsupported_question requires the unsupported intent"

    # scope identity
    if scope == "assessment":
        if not context:
            return False, "assessment scope requires exactly one retrieved snapshot"
        expected_id = str(context[0].assessment_id)
        if output.assessment_id != expected_id:
            return False, "assessment_id does not match the requested assessment"
        if output.machine_id is not None:
            return False, "an assessment-scoped answer must not carry a machine_id"
        if output.lookback_days is not None:
            return False, "an assessment-scoped answer must not carry a lookback window"
        if output.window_start is not None or output.window_end is not None:
            return False, "an assessment-scoped answer must not carry a window"
        if output.history_truncated:
            return False, "an assessment-scoped answer cannot be truncated"
        if output.referenced_assessment_ids != [expected_id]:
            return False, "referenced_assessment_ids must contain only the requested assessment"
    else:
        if output.machine_id != machine_id:
            return False, "machine_id does not match the requested machine"
        if output.assessment_id is not None:
            return False, "a machine-scoped answer must not carry an assessment_id"
        if output.lookback_days != lookback_days:
            return False, "lookback_days does not match the resolved window"
        expected_end = context[0].created_at if context else None
        expected_start = expected_end - timedelta(days=lookback_days) if context else None
        if _norm(output.window_start) != _norm(expected_start) or _norm(output.window_end) != _norm(expected_end):
            return False, "window does not match the stored created_at range"
        if output.history_truncated != history_truncated:
            return False, "history_truncated does not match the resolved retrieval"
        ids = [str(snapshot.assessment_id) for snapshot in context]
        if sorted(output.referenced_assessment_ids) != sorted(ids) or len(
            output.referenced_assessment_ids
        ) != len(ids):
            return False, "referenced_assessment_ids must be exactly the retrieved window"

    if output.assessment_count != len(context):
        return False, "assessment_count does not match the retrieved context"

    stored_evidence = {str(item.evidence_id) for snapshot in context for item in snapshot.evidence}
    if not set(output.referenced_evidence_ids) <= stored_evidence:
        return False, "referenced_evidence_ids contain ids that are not stored in the context"
    if len(set(output.referenced_evidence_ids)) != len(output.referenced_evidence_ids):
        return False, "referenced_evidence_ids contain duplicates"

    return _ground_text(output, context, machine_id=machine_id, lookback_days=lookback_days)
