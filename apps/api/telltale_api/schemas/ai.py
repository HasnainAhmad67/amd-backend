"""Central AI Analysis Agent contracts - slice A: AI Result Explanation only.

Strict, additive schemas: every model uses ``extra="forbid"`` so a client can never
smuggle authoritative fields (statuses, ids, scores, timestamps) through an AI
request or an AI record. These contracts describe what the agent MAY say about an
immutable assessment snapshot; they never extend or reinterpret the assessment
contracts themselves (``schemas/assessment.py`` stays untouched).

Rules encoded here:
- ``validation_status`` is one of validated / fallback / refused. An AI output is
  never persisted as "validated" unless the grounding validator accepted it.
- ``referenced_assessment_ids`` / ``referenced_evidence_ids`` are the ONLY ids the
  agent may cite; the validator checks each one against the snapshot.
- Bounded lengths everywhere: no unbounded AI text is ever stored or returned.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .assessment import ModalityKind, OverallStatus, QualityCode, SourceMode

PROMPT_VERSION_EXPLAIN = "explain/v1"
PROMPT_VERSION_REVIEW = "evidence-review/v1"
PROMPT_VERSION_ANALYZE = "analyze/v1"
PROMPT_VERSION_REPORT = "report/v1"
PROMPT_VERSION_ASK = "ask/v1"
# Kept for backwards compatibility with slice A imports.
PROMPT_VERSION = PROMPT_VERSION_EXPLAIN

ValidationStatus = Literal["validated", "fallback", "refused"]
AiKind = Literal["explain", "evidence_review", "analyze", "report", "ask"]
AiFeature = Literal["explain", "evidence_review", "analyze", "report", "ask"]
AgreementValue = Literal["consistent_indications", "mixed_indications", "indeterminate"]

# Slice E (Ask Telltale): fixed, server-owned question scopes, supported intents and
# refusal reasons. No free-form reasoning: the question is inert data matched against
# a fixed dictionary.
AskScope = Literal["assessment", "machine"]
AskIntent = Literal[
    "status",
    "modalities",
    "flags_uncertainty",
    "sensor",
    "acoustic",
    "bearing",
    "provenance",
    "history",
    "unsupported",
]
AskRefusalReason = Literal[
    "unsupported_question",
    "evidence_insufficient",
    "no_evidence",
    "scope_mismatch",
]

MAX_EXPLANATION_CHARS = 8000
MAX_NOTE_CHARS = 800
MAX_IDS = 50  # also the report's / ask's maximum referenced assessments
MAX_REPORT_EVIDENCE = 250
MAX_QUESTION_CHARS = 500  # the caller's question is required and bounded
MAX_QUESTION_ECHO_CHARS = 300  # the stored echo is a bounded, sanitized copy


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AiExplainRequest(_Strict):
    """Request contract: exactly one assessment id, nothing else."""

    assessment_id: str = Field(min_length=1, max_length=64)


class AiModalityNote(_Strict):
    """What the agent may state about one modality - facts quoted from the snapshot."""

    kind: ModalityKind
    availability: str
    reason_code: str
    source_mode: SourceMode
    observed_at: datetime | None = None
    inference_at: datetime | None = None
    model_name: str | None = None
    model_version: str | None = None
    training_dataset: str | None = None
    evidence_ids: list[str] = Field(default_factory=list, max_length=MAX_IDS)
    facts: list[str] = Field(default_factory=list, max_length=12)


class AiExplainOutput(_Strict):
    """Structured, grounded explanation. Numeric values are STORED values only."""

    feature: Literal["explain"] = "explain"
    assessment_id: str
    machine_id: str
    overall_status: OverallStatus
    status_reasons: list[str] = Field(default_factory=list, max_length=20)
    expected_modalities: list[ModalityKind] = Field(max_length=5)
    present_modalities: list[ModalityKind] = Field(max_length=5)
    completeness_note: str = Field(max_length=MAX_NOTE_CHARS)
    uncertainty_notes: list[str] = Field(default_factory=list, max_length=12)
    per_modality: list[AiModalityNote] = Field(default_factory=list, max_length=5)
    quality_flag_codes: list[QualityCode] = Field(default_factory=list, max_length=20)
    limitations: list[str] = Field(default_factory=list, max_length=50)
    explanation_text: str = Field(min_length=120, max_length=MAX_EXPLANATION_CHARS)
    referenced_assessment_ids: list[str] = Field(default_factory=list, max_length=5)
    referenced_evidence_ids: list[str] = Field(default_factory=list, max_length=MAX_IDS)


class AiEvidenceReviewRequest(_Strict):
    """Request contract: exactly one assessment id, nothing else."""

    assessment_id: str = Field(min_length=1, max_length=64)


class AiModalityReview(_Strict):
    """Per-modality review row: only fields that exist in the stored snapshot."""

    kind: ModalityKind
    availability: str
    reason_code: str
    source_mode: SourceMode
    observed_at: datetime | None = None
    inference_at: datetime | None = None
    model_name: str | None = None
    model_version: str | None = None
    training_dataset: str | None = None
    evidence_kinds: list[str] = Field(default_factory=list, max_length=20)
    limitations: list[str] = Field(default_factory=list, max_length=50)


class AiFlagEntry(_Strict):
    """A stored quality flag (code + severity), never re-derived or invented."""

    code: str = Field(max_length=64)
    severity: str = Field(max_length=16)


class AiEvidenceReviewOutput(_Strict):
    """Structured evidence & uncertainty review. Facts-only; never a fault claim."""

    feature: Literal["evidence_review"]
    assessment_id: str
    machine_id: str
    overall_status: OverallStatus
    status_reasons: list[str] = Field(default_factory=list, max_length=20)
    expected_modalities: list[ModalityKind] = Field(max_length=5)
    present_modalities: list[ModalityKind] = Field(max_length=5)
    missing_modalities: list[ModalityKind] = Field(max_length=5)
    completeness_note: str = Field(max_length=MAX_NOTE_CHARS)
    modality_reviews: list[AiModalityReview] = Field(default_factory=list, max_length=5)
    quality_flags: list[AiFlagEntry] = Field(default_factory=list, max_length=20)
    uncertainty_indicators: list[str] = Field(default_factory=list, max_length=20)
    provenance_summary: str = Field(max_length=MAX_NOTE_CHARS)
    non_claims: list[str] = Field(default_factory=list, max_length=10)
    review_text: str = Field(min_length=120, max_length=MAX_EXPLANATION_CHARS)
    referenced_assessment_ids: list[str] = Field(default_factory=list, max_length=5)
    referenced_evidence_ids: list[str] = Field(default_factory=list, max_length=MAX_IDS)


class AiAnalyzeRequest(_Strict):
    """Request contract: exactly one assessment id, nothing else."""

    assessment_id: str = Field(min_length=1, max_length=64)


class AiModalitySummary(_Strict):
    """Per-modality factual summary: only fields that exist in the stored snapshot."""

    kind: ModalityKind
    availability: str
    reason_code: str
    source_mode: SourceMode
    evidence_kinds: list[str] = Field(default_factory=list, max_length=20)
    limitations: list[str] = Field(default_factory=list, max_length=50)
    indication: str = Field(max_length=400)


class AiPairRelation(_Strict):
    """One pairwise relation between two AVAILABLE modalities (stored tokens only)."""

    kinds: list[ModalityKind] = Field(max_length=2)
    relation: str = Field(max_length=600)


class AiAnalyzeOutput(_Strict):
    """Structured cross-modality analysis. Facts-only; never a fused score."""

    feature: Literal["analyze"]
    assessment_id: str
    machine_id: str
    overall_status: OverallStatus
    status_reasons: list[str] = Field(default_factory=list, max_length=20)
    expected_modalities: list[ModalityKind] = Field(max_length=5)
    present_modalities: list[ModalityKind] = Field(max_length=5)
    missing_modalities: list[ModalityKind] = Field(max_length=5)
    completeness_note: str = Field(max_length=MAX_NOTE_CHARS)
    modality_summaries: list[AiModalitySummary] = Field(default_factory=list, max_length=5)
    relations: list[AiPairRelation] = Field(default_factory=list, max_length=3)
    agreement: AgreementValue
    agreement_note: str = Field(max_length=MAX_NOTE_CHARS)
    conservative_interpretation: list[str] = Field(default_factory=list, max_length=10)
    uncertainty_indicators: list[str] = Field(default_factory=list, max_length=20)
    provenance_summary: str = Field(max_length=MAX_NOTE_CHARS)
    non_claims: list[str] = Field(default_factory=list, max_length=10)
    analysis_text: str = Field(min_length=120, max_length=MAX_EXPLANATION_CHARS)
    referenced_assessment_ids: list[str] = Field(default_factory=list, max_length=5)
    referenced_evidence_ids: list[str] = Field(default_factory=list, max_length=MAX_IDS)


class AiReportRequest(_Strict):
    """Request contract: one machine id plus an optional bounded lookback window.

    Nothing else is accepted (``extra="forbid"``): scores, statuses, ids, timestamps
    and the assessment selection itself are server-owned. The window is expressed in
    days and is resolved server-side against stored ``created_at`` values only.
    """

    machine_id: str = Field(min_length=1, max_length=128)
    lookback_days: int = Field(default=30, ge=1, le=90)


class AiAvailabilityCount(_Strict):
    """One stored availability state and how many modality records carry it."""

    state: str = Field(max_length=32)
    count: int = Field(ge=0)


class AiModalityAvailability(_Strict):
    """Per-modality availability counts and the distinct stored reason codes."""

    kind: ModalityKind
    counts: list[AiAvailabilityCount] = Field(default_factory=list, max_length=5)
    reason_codes: list[str] = Field(default_factory=list, max_length=10)


class AiStatusCount(_Strict):
    """Stored overall-status distribution entry (never a fused or health score)."""

    status: OverallStatus
    count: int = Field(ge=0)


class AiRecurringFlag(_Strict):
    """A stored quality flag (code + severity) with how often it recurred."""

    code: str = Field(max_length=64)
    severity: str = Field(max_length=16)
    count: int = Field(ge=1)


class AiReviewSuggestion(_Strict):
    """A NON-BINDING review suggestion derived only from stored evidence patterns."""

    scope: Literal["review_suggestion"] = "review_suggestion"
    title: str = Field(min_length=1, max_length=200)
    basis: str = Field(min_length=1, max_length=300)
    detail: str = Field(min_length=1, max_length=600)


class AiReportOutput(_Strict):
    """Structured, facts-only maintenance report over stored assessment history.

    Every field is derived from stored snapshot values; no score, p-value, confidence,
    status, id or timestamp is created, and no fused failure probability, health score
    or risk score is produced. ``window_start``/``window_end`` are a RANGE derived from
    stored ``created_at`` values (never a fabricated measurement timestamp).
    """

    feature: Literal["report"] = "report"
    machine_id: str = Field(min_length=1, max_length=128)
    window_start: datetime | None = None
    window_end: datetime | None = None
    assessment_count: int = Field(ge=0, le=MAX_IDS)
    history_truncated: bool = False
    referenced_assessment_ids: list[str] = Field(default_factory=list, max_length=MAX_IDS)
    modality_availability: list[AiModalityAvailability] = Field(default_factory=list, max_length=3)
    status_distribution: list[AiStatusCount] = Field(default_factory=list, max_length=4)
    recurring_flags: list[AiRecurringFlag] = Field(default_factory=list, max_length=20)
    missing_data_patterns: list[str] = Field(default_factory=list, max_length=20)
    conservative_observations: list[str] = Field(default_factory=list, max_length=20)
    review_suggestions: list[AiReviewSuggestion] = Field(default_factory=list, max_length=10)
    limitations: list[str] = Field(default_factory=list, max_length=50)
    uncertainty_notes: list[str] = Field(default_factory=list, max_length=20)
    non_claims: list[str] = Field(default_factory=list, max_length=10)
    report_text: str = Field(min_length=120, max_length=MAX_EXPLANATION_CHARS)
    referenced_evidence_ids: list[str] = Field(default_factory=list, max_length=MAX_REPORT_EVIDENCE)


class AiOutputRecord(_Strict):
    """Append-only AI output record (stored + returned by GET). Immutable once written."""

    feature: AiFeature = "explain"
    ai_output_id: UUID
    owner: str
    kind: AiKind
    assessment_ids: list[str] = Field(max_length=MAX_IDS)
    provider: str = Field(max_length=32)
    model: str | None = Field(default=None, max_length=64)
    prompt_version: str = Field(max_length=32)
    prompt_sha256: str = Field(pattern=r"[0-9a-f]{64}")
    # Union, not a discriminator: rows stored before a newer feature was added still
    # validate through the default feature, and each payload shape only matches its
    # own member (strict extra="forbid" + feature-specific required fields).
    output: AiExplainOutput | AiEvidenceReviewOutput | AiAnalyzeOutput | AiReportOutput | AiAskOutput
    response_text: str = Field(max_length=MAX_EXPLANATION_CHARS)
    validation_status: ValidationStatus
    created_at: datetime


class AiExplainResponse(_Strict):
    """201 response of POST /api/v1/ai/explain (safe subset of the stored record)."""

    feature: Literal["explain"] = "explain"
    ai_output_id: UUID
    assessment_id: str
    kind: AiKind
    explanation_text: str = Field(max_length=MAX_EXPLANATION_CHARS)
    validation_status: ValidationStatus
    provider: str = Field(max_length=32)
    model: str | None = Field(default=None, max_length=64)
    prompt_version: str = Field(max_length=32)
    referenced_assessment_ids: list[str] = Field(max_length=5)
    created_at: datetime


class AiEvidenceReviewResponse(_Strict):
    """201 response of POST /api/v1/ai/evidence-review (safe subset of the record)."""

    feature: Literal["evidence_review"]
    ai_output_id: UUID
    assessment_id: str
    kind: AiKind
    review_text: str = Field(max_length=MAX_EXPLANATION_CHARS)
    validation_status: ValidationStatus
    provider: str = Field(max_length=32)
    model: str | None = Field(default=None, max_length=64)
    prompt_version: str = Field(max_length=32)
    referenced_assessment_ids: list[str] = Field(max_length=5)
    created_at: datetime


class AiAnalyzeResponse(_Strict):
    """201 response of POST /api/v1/ai/analyze (safe subset of the stored record)."""

    feature: Literal["analyze"]
    ai_output_id: UUID
    assessment_id: str
    kind: AiKind
    analysis_text: str = Field(max_length=MAX_EXPLANATION_CHARS)
    validation_status: ValidationStatus
    provider: str = Field(max_length=32)
    model: str | None = Field(default=None, max_length=64)
    prompt_version: str = Field(max_length=32)
    referenced_assessment_ids: list[str] = Field(max_length=5)
    created_at: datetime


class AiReportResponse(_Strict):
    """201 response of POST /api/v1/ai/report (safe subset of the stored record)."""

    feature: Literal["report"] = "report"
    ai_output_id: UUID
    machine_id: str = Field(min_length=1, max_length=128)
    kind: AiKind
    window_start: datetime | None = None
    window_end: datetime | None = None
    assessment_count: int = Field(ge=0, le=MAX_IDS)
    history_truncated: bool = False
    report_text: str = Field(max_length=MAX_EXPLANATION_CHARS)
    validation_status: ValidationStatus
    provider: str = Field(max_length=32)
    model: str | None = Field(default=None, max_length=64)
    prompt_version: str = Field(max_length=32)
    referenced_assessment_ids: list[str] = Field(max_length=MAX_IDS)
    created_at: datetime


class AiAskRequest(_Strict):
    """Request contract for POST /ai/ask: one bounded question and exactly one scope.

    The question is UNTRUSTED inert data (never an instruction). Exactly one of
    ``assessment_id`` / ``machine_id`` is required, and ``lookback_days`` is allowed
    only with ``machine_id`` - those scope rules are enforced by the route so that the
    precise error codes (``scope_required`` / ``scope_mismatch``) can be returned.
    """

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    assessment_id: str | None = Field(default=None, max_length=64)
    machine_id: str | None = Field(default=None, max_length=128)
    lookback_days: int | None = Field(default=None, ge=1, le=90)


class AiAskOutput(_Strict):
    """Structured, facts-only answer over owner-visible immutable assessment evidence.

    Every field derives from stored snapshot values; the question is echoed only as a
    bounded sanitized digest/copy and is NEVER interpreted as an instruction. No score,
    p-value, confidence, status, id or timestamp is created, and no fused failure
    probability, health score or risk score is produced.
    """

    feature: Literal["ask"] = "ask"
    scope: AskScope
    assessment_id: str | None = None
    machine_id: str | None = None
    lookback_days: int | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    history_truncated: bool = False
    assessment_count: int = Field(ge=0, le=MAX_IDS)
    question_sha256: str = Field(pattern=r"[0-9a-f]{64}")
    question_echo: str = Field(max_length=MAX_QUESTION_ECHO_CHARS)
    intent: AskIntent
    matched_terms: list[str] = Field(default_factory=list, max_length=8)
    answered: bool
    refusal_reason: AskRefusalReason | None = None
    relevant_facts: list[str] = Field(default_factory=list, max_length=12)
    present_modalities: list[ModalityKind] = Field(default_factory=list, max_length=3)
    uncertainty_indicators: list[str] = Field(default_factory=list, max_length=12)
    limitations: list[str] = Field(default_factory=list, max_length=50)
    non_claims: list[str] = Field(default_factory=list, max_length=10)
    referenced_assessment_ids: list[str] = Field(default_factory=list, max_length=MAX_IDS)
    referenced_evidence_ids: list[str] = Field(
        default_factory=list, max_length=MAX_REPORT_EVIDENCE
    )
    answer_summary: str = Field(min_length=1, max_length=MAX_NOTE_CHARS)
    answer_text: str = Field(min_length=120, max_length=MAX_EXPLANATION_CHARS)


class AiAskResponse(_Strict):
    """201 response of POST /api/v1/ai/ask (safe subset of the stored record)."""

    feature: Literal["ask"] = "ask"
    ai_output_id: UUID
    kind: AiKind
    scope: AskScope
    question_sha256: str = Field(pattern=r"[0-9a-f]{64}")
    question_echo: str = Field(max_length=MAX_QUESTION_ECHO_CHARS)
    answered: bool
    refusal_reason: AskRefusalReason | None = None
    answer_summary: str = Field(max_length=MAX_NOTE_CHARS)
    answer_text: str = Field(max_length=MAX_EXPLANATION_CHARS)
    validation_status: ValidationStatus
    provider: str = Field(max_length=32)
    model: str | None = Field(default=None, max_length=64)
    prompt_version: str = Field(max_length=32)
    referenced_assessment_ids: list[str] = Field(max_length=MAX_IDS)
    created_at: datetime
