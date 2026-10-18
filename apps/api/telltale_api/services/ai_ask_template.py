"""Central AI Analysis Agent - slice E: Ask Telltale (deterministic, offline).

Answers a user question from owner-visible immutable assessment evidence only. This
slice does NOT provide free-form natural-language reasoning: the question is treated
as UNTRUSTED inert data (never an instruction), sanitized and normalized only for
matching against a FIXED server-owned supported-intent dictionary, and answered with
fixed templates built from stored snapshot facts. If no supported intent matches, or
the retrieved context does not store the evidence an intent needs, a deterministic
refusal is produced - never a speculative answer.

Reuses the slice-A facts-only projection boundary (``ai_template``) and the slice-D
aggregate helpers (``ai_report_template``), so it never calls a model, never reads raw
audio/telemetry/paths and never creates a score, p-value, confidence, status, id or
timestamp. No fused failure probability, health score or risk score is ever computed.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any

from ..schemas.ai import (
    MAX_EXPLANATION_CHARS,
    MAX_NOTE_CHARS,
    MAX_QUESTION_ECHO_CHARS,
    PROMPT_VERSION_ASK,
    AiAskOutput,
)
from ..schemas.assessment import Assessment
from .ai_report_template import modality_availability, recurring_flags, status_distribution
from .ai_template import SYSTEM_INSTRUCTIONS, _clean, _iso, _modality_facts, project_assessment, uncertaintynotes

CANONICAL_KINDS = ("sensor", "acoustic", "bearing")

# Fixed non-claims, identical on every answer (structured field + text footer).
NON_CLAIMS = [
    "This answer is not a fault diagnosis and does not confirm any fault.",
    "This answer is not machine-health completeness and makes no coverage claim.",
    "This answer is not coverage of machine health.",
    "This answer is not a fused failure probability, health score or risk score.",
    "This answer is not a guaranteed maintenance plan.",
    "This answer is limited to supported evidence topics and does not provide free-form "
    "natural-language reasoning.",
]

FOOTER = (
    "This answer quotes stored assessment facts for supported topics only. It is not a fault "
    "diagnosis, not machine-health completeness, and not coverage; it creates no fused failure "
    "probability, health score or risk score, and it is not a guaranteed maintenance plan. It "
    "does not provide free-form natural-language reasoning beyond these supported evidence topics."
)

REQUIRED_PHRASES = (
    "not a fault diagnosis",
    "not machine-health completeness",
    "not coverage",
    "not a fused failure probability",
    "not a guaranteed maintenance plan",
    "free-form natural-language reasoning",
)

# Wording the agent must never emit: confirmed faults, guaranteed fixes, prescribed work.
FORBIDDEN_PHRASES = (
    "definitely faulty",
    "confirmed diagnosis",
    "will fix",
    "will prevent",
    "maintenance is required",
    "guaranteed to",
    "machine is healthy",
    "no maintenance needed",
)

# Fixed priority order: the first intent with a matching canonical term wins. Modality
# topics are more specific than generic status, which is more specific than history.
INTENT_ORDER = (
    "sensor",
    "acoustic",
    "bearing",
    "status",
    "modalities",
    "flags_uncertainty",
    "provenance",
    "history",
)

# Server-owned dictionary. Only these canonical terms may ever appear in matched_terms.
INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sensor": (
        "sensor", "temperature", "torque", "speed", "rotational", "tool", "wear",
        "air", "process", "probability", "failure",
    ),
    "acoustic": (
        "acoustic", "sound", "audio", "anomaly", "anomalous", "noise",
        "conformal", "spectrogram",
    ),
    "bearing": (
        "bearing", "vibration", "fault", "class", "classifier", "prediction",
        "race", "ball", "defect",
    ),
    "status": (
        "status", "condition", "state", "health", "overall", "risk", "severity",
        "critical", "normal", "watch",
    ),
    "modalities": (
        "modality", "modalities", "available", "availability", "evidence",
        "present", "missing", "submitted",
    ),
    "flags_uncertainty": (
        "flag", "flags", "warning", "warnings", "uncertain", "uncertainty",
        "limitation", "limitations", "caveat", "caveats", "reliable", "trust",
    ),
    "provenance": (
        "provenance", "source", "sources", "model", "models", "dataset",
        "simulated", "uploaded", "demo", "identity", "version", "trained",
    ),
    "history": (
        "history", "count", "many", "assessments", "records", "previous",
        "recent", "window", "lookback",
    ),
}

# The complete canonical dictionary (used by the validator).
ALL_KEYWORDS: frozenset[str] = frozenset(
    term for terms in INTENT_KEYWORDS.values() for term in terms
)

# Human-readable supported topics for the unsupported-question refusal.
SUPPORTED_TOPICS = (
    "status",
    "modalities",
    "flags and uncertainty",
    "sensor",
    "acoustic",
    "bearing",
    "provenance",
    "machine history",
)

_WORD = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_question(text: str) -> str:
    """Lowercase, drop anything that is not a letter/digit, collapse whitespace.

    Deterministic and inert: the result is only used for dictionary matching."""
    lowered = text.lower()
    spaced = _NON_ALNUM.sub(" ", lowered)
    return _WORD.sub(" ", spaced).strip()


def question_digest(normalized: str) -> str:
    """sha256 of the normalized question (the raw question is never stored)."""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def sanitize_echo(text: str) -> str:
    """Bounded, display-safe copy of the caller's question (control chars stripped,
    whitespace collapsed, double quotes neutralized). Never interpreted, never used
    as evidence, and excluded from evidence-grounding scans."""
    printable = "".join(ch if ch.isprintable() else " " for ch in text)
    flat = _WORD.sub(" ", printable).strip().replace('"', "'")
    if len(flat) > MAX_QUESTION_ECHO_CHARS:
        flat = flat[: MAX_QUESTION_ECHO_CHARS - 3].rstrip() + "..."
    return flat


def classify_question(normalized: str) -> tuple[str, list[str]]:
    """Map a normalized question to a supported intent via the fixed dictionary.

    Returns (intent, matched_terms). ``intent`` is "unsupported" with no terms when
    nothing matches. ``matched_terms`` contains ONLY canonical dictionary terms."""
    tokens = set(normalized.split())
    for intent in INTENT_ORDER:
        matched = [term for term in INTENT_KEYWORDS[intent] if term in tokens]
        if matched:
            return intent, matched[:8]
    return "unsupported", []


def _primary_token(kind: str, assessment: Assessment) -> str | None:
    """The stored primary indication token for one available modality (or None)."""
    for record in assessment.modalities:
        if record.kind != kind or record.availability != "available" or record.output is None:
            continue
        output = record.output.model_dump(mode="json")
        if kind == "sensor":
            value = output.get("risk_level")
        elif kind == "acoustic":
            value = output.get("status")
        else:
            value = output.get("predicted_class")
        return str(value) if isinstance(value, str) else None
    return None


def _facts_status(context: list[Assessment], scope: str) -> tuple[list[str], bool]:
    if scope == "assessment":
        snapshot = context[0]
        completeness = project_assessment(snapshot)["evidence_completeness"]
        facts = [
            f'Stored overall status is "{snapshot.overall_status}".',
            f"{completeness['present']} of {completeness['expected']} requested modalities "
            "produced available evidence in this assessment.",
        ]
        return facts, False
    distribution = status_distribution(context)
    parts = ", ".join(f'"{row.status}" {row.count}' for row in distribution) or "none stored"
    return [
        f"{len(context)} owner-visible assessments are inside the requested window.",
        f"Stored overall-status distribution across the window: {parts}.",
    ], False


def _facts_modalities(context: list[Assessment]) -> tuple[list[str], bool]:
    facts: list[str] = []
    for row in modality_availability(context):
        counts = ", ".join(f'"{count.state}" {count.count}' for count in row.counts) or "none stored"
        facts.append(f"{row.kind}: availability {counts}.")
        if row.reason_codes:
            reasons = ", ".join(f'"{code}"' for code in row.reason_codes)
            facts.append(f"{row.kind}: stored reason codes {reasons}.")
    return (facts[:12] or ["No modality availability was stored."]), False


def _facts_flags(context: list[Assessment]) -> tuple[list[str], bool]:
    flags = recurring_flags(context)
    if not flags:
        return ["No stored quality flags recur in the retrieved evidence."], False
    parts = "; ".join(f'"{flag.code}" ({flag.severity}) x{flag.count}' for flag in flags)
    return [f"Recurring stored quality flags: {parts}."], False


def _facts_kind(context: list[Assessment], scope: str, kind: str) -> tuple[list[str], bool]:
    if scope == "assessment":
        snapshot = context[0]
        for record in snapshot.modalities:
            if record.kind == kind and record.availability == "available" and record.output is not None:
                return _modality_facts(kind, record.output.model_dump(mode="json")), False
        return [], True
    available = [
        snapshot
        for snapshot in context
        if any(m.kind == kind and m.availability == "available" for m in snapshot.modalities)
    ]
    if not available:
        return [], True
    facts = [
        f"{len(available)} of {len(context)} assessments in the window have the {kind} "
        "modality available."
    ]
    tokens = sorted({token for snapshot in available if (token := _primary_token(kind, snapshot))})
    if tokens:
        facts.append(f"{kind} stored indications observed: " + ", ".join(f'"{t}"' for t in tokens) + ".")
    return facts, False


def _facts_provenance(context: list[Assessment]) -> tuple[list[str], bool]:
    facts: list[str] = []
    sources = sorted({snapshot.source_mode for snapshot in context})
    facts.append("Assessment source modes observed: " + ", ".join(f'"{s}"' for s in sources) + ".")
    modality_sources = sorted({m.source_mode for snapshot in context for m in snapshot.modalities})
    facts.append(
        "Modality source modes observed: " + ", ".join(f'"{s}"' for s in modality_sources) + "."
    )
    seen: set[tuple] = set()
    for snapshot in context:
        for record in snapshot.modalities:
            identity = record.model_identity
            if identity is None:
                continue
            key = (record.kind, identity.name, identity.version, identity.training_dataset)
            if key in seen:
                continue
            seen.add(key)
            facts.append(
                f"{record.kind} model: name {identity.name or 'unnamed model'}, version "
                f"{identity.version or 'no version'}, trained on "
                f"{identity.training_dataset or 'no training dataset recorded'}."
            )
    return facts[:12], False


def _facts_history(
    context: list[Assessment], *, window_start: datetime | None, window_end: datetime | None,
    history_truncated: bool,
) -> tuple[list[str], bool]:
    return [
        f"{len(context)} owner-visible assessments are inside the requested window.",
        f"Window start {_iso(window_start)} and window end {_iso(window_end)}, both derived from "
        "stored created_at values only.",
        f'History truncated: {"true" if history_truncated else "false"}.',
    ], False


def _uncertainty(context: list[Assessment], *, history_truncated: bool) -> list[str]:
    notes: list[str] = []
    if any(m.availability != "available" for snapshot in context for m in snapshot.modalities):
        notes.append(
            "Some requested modalities did not produce available evidence; missing data limits "
            "interpretation and was not fabricated."
        )
    codes = {flag.code for snapshot in context for flag in snapshot.quality_flags}
    if "TIMESTAMP_MISSING" in codes:
        notes.append("A measurement timestamp is missing; no age or freshness judgement is made.")
    if "SOURCE_ILLUSTRATIVE" in codes:
        notes.append("At least one input is illustrative demo/simulated data, not a real-machine measurement.")
    if "MIXED_SOURCE_PROVENANCE" in codes:
        notes.append("Available evidence combines different source modes; the sources are unrelated.")
    if "BEARING_UNCERTAIN" in codes:
        notes.append(
            "Stored bearing classifier uncertainty limits interpretation; a conservative review "
            "status is reported, not a confirmed fault."
        )
    if history_truncated:
        notes.append("History was truncated to the newest assessments in the window.")
    return uncertaintynotes(notes)


def _limitations(context: list[Assessment]) -> list[str]:
    limits: list[str] = []
    seen: set[str] = set()
    for snapshot in context:
        for record in snapshot.modalities:
            for item in record.limitations:
                flat = _clean(item, 600, quotes=False)
                if flat not in seen:
                    seen.add(flat)
                    limits.append(flat)
    limits.append("Only stored assessments were read; no model was re-run and no telemetry or audio was accessed.")
    limits.append(
        "This answer covers the retrieved owner-visible evidence only; it is not machine-health "
        "completeness or coverage."
    )
    return limits[:50]


class TemplateAskProvider:
    """Deterministic offline Ask Telltale provider (no network, no ML, no clock)."""

    provider = "template"
    model: str | None = None

    def ask(
        self,
        context: list[Assessment],
        *,
        scope: str,
        question: str,
        lookback_days: int | None,
        window_start: datetime | None,
        window_end: datetime | None,
        history_truncated: bool,
        machine_id: str | None,
    ) -> AiAskOutput:
        normalized = normalize_question(question)
        intent, matched = classify_question(normalized)
        ids = [str(snapshot.assessment_id) for snapshot in context]
        evidence_ids = [str(item.evidence_id) for snapshot in context for item in snapshot.evidence][
            :250
        ]
        present = [
            kind
            for kind in CANONICAL_KINDS
            if any(m.kind == kind and m.availability == "available" for s in context for m in s.modalities)
        ]
        label = (
            f"machine {machine_id}" if scope == "machine"
            else f"assessment {ids[0]} for machine {context[0].machine_id}"
        )
        common: dict[str, Any] = {
            "scope": scope,
            "assessment_id": ids[0] if scope == "assessment" else None,
            "machine_id": machine_id if scope == "machine" else None,
            "lookback_days": lookback_days,
            "window_start": window_start,
            "window_end": window_end,
            "history_truncated": history_truncated,
            "assessment_count": len(context),
            "question_sha256": question_digest(normalized),
            "question_echo": sanitize_echo(question),
            "present_modalities": present,
            "referenced_assessment_ids": ids,
            "referenced_evidence_ids": evidence_ids,
        }

        # ---- deterministic refusals (checked in a fixed order) --------------------
        if intent == "unsupported":
            return self._refusal(
                common, context, label=label, intent="unsupported", matched=[],
                reason="unsupported_question",
                summary="The question did not match any supported evidence topic.",
                detail=(
                    "The question could not be matched to a supported evidence topic, so no answer "
                    f"is offered. Supported topics: {', '.join(SUPPORTED_TOPICS)}."
                ),
                limitations=[], history_truncated=history_truncated,
            )
        if scope == "machine" and not context:
            return self._refusal(
                common, context, label=label, intent=intent, matched=matched, reason="no_evidence",
                summary="No owner-visible assessments were stored inside the window.",
                detail=(
                    f"No owner-visible assessments were found for machine {machine_id} within the "
                    f"requested {lookback_days}-day window, so this question cannot be answered from "
                    "stored evidence and nothing was fabricated."
                ),
                limitations=[], history_truncated=history_truncated,
            )
        if intent == "history" and scope == "assessment":
            return self._refusal(
                common, context, label=label, intent=intent, matched=matched, reason="scope_mismatch",
                summary="The machine-history topic needs a machine scope, not an assessment scope.",
                detail=(
                    "The matched topic is available only for a machine window, and this request used "
                    "an assessment scope, so no answer is offered."
                ),
                limitations=_limitations(context), history_truncated=history_truncated,
            )

        # ---- build the supported answer -------------------------------------------
        if intent == "status":
            facts, missing = _facts_status(context, scope)
        elif intent == "modalities":
            facts, missing = _facts_modalities(context)
        elif intent == "flags_uncertainty":
            facts, missing = _facts_flags(context)
        elif intent in ("sensor", "acoustic", "bearing"):
            facts, missing = _facts_kind(context, scope, intent)
        elif intent == "provenance":
            facts, missing = _facts_provenance(context)
        else:  # history (machine scope only, already guarded)
            facts, missing = _facts_history(
                context, window_start=window_start, window_end=window_end,
                history_truncated=history_truncated,
            )

        if missing:
            return self._refusal(
                common, context, label=label, intent=intent, matched=matched,
                reason="evidence_insufficient",
                summary=f"The stored evidence does not cover the {intent} topic for this scope.",
                detail=(
                    f"The question matched the {intent} topic, but the retrieved context does not "
                    "store the evidence needed to answer it, so no answer is offered and nothing "
                    "was fabricated."
                ),
                limitations=_limitations(context), history_truncated=history_truncated,
            )

        uncertainty = _uncertainty(context, history_truncated=history_truncated)
        limitations = _limitations(context)
        summary = f"Answered the supported {intent} topic from stored assessment facts."
        text = self._compose(
            label, intent, matched, summary, facts, uncertainty, limitations
        )
        return AiAskOutput(
            feature="ask", intent=intent, matched_terms=matched, answered=True,
            refusal_reason=None, relevant_facts=facts[:12],
            uncertainty_indicators=uncertainty, limitations=limitations,
            non_claims=list(NON_CLAIMS), answer_summary=summary[:MAX_NOTE_CHARS],
            answer_text=text, **common,
        )

    def _refusal(
        self,
        common: dict[str, Any],
        context: list[Assessment],
        *,
        label: str,
        intent: str,
        matched: list[str],
        reason: str,
        summary: str,
        detail: str,
        limitations: list[str],
        history_truncated: bool,
    ) -> AiAskOutput:
        uncertainty = _uncertainty(context, history_truncated=history_truncated) or [
            "No interpretation is offered because the evidence required for this topic is not stored."
        ]
        text = self._compose(label, intent, matched, summary, [detail], uncertainty, limitations)
        return AiAskOutput(
            feature="ask", intent=intent, matched_terms=matched, answered=False,
            refusal_reason=reason, relevant_facts=[detail],
            uncertainty_indicators=uncertainty, limitations=limitations,
            non_claims=list(NON_CLAIMS), answer_summary=summary[:MAX_NOTE_CHARS],
            answer_text=text, **common,
        )

    def _compose(
        self,
        label: str,
        intent: str,
        matched: list[str],
        summary: str,
        facts: list[str],
        uncertainty: list[str],
        limitations: list[str],
    ) -> str:
        """Compose a concise, evidence-grounded answer for the user."""
        sections = [
            summary.strip(),
            *[fact.strip() for fact in facts if fact.strip()],
        ]

        if uncertainty:
            sections.append(
                "Uncertainty: " + "; ".join(
                    item.strip() for item in uncertainty if item.strip()
                )
            )

        if limitations:
            sections.append(
                "Limitations: " + "; ".join(
                    item.strip() for item in limitations if item.strip()
                )
            )

        head = _clean(" ".join(section for section in sections if section))
        tail = _clean(_tail_text())

        if not head:
            head = (
                "There is not enough stored evidence to provide "
                "a reliable answer."
            )

        budget = max(0, MAX_EXPLANATION_CHARS - len(tail) - 1)
        return (head[:budget].rstrip() + " " + tail)[:MAX_EXPLANATION_CHARS]    
    def prompt_sha256(
        self,
        context: list[Assessment],
        *,
        scope: str,
        question: str,
        lookback_days: int | None,
        machine_id: str | None,
    ) -> str:
        """Deterministic digest of the effective prompt (version + instructions + facts +
        the normalized question). The raw question itself is never stored."""
        canonical = json.dumps(
            {
                "prompt_version": PROMPT_VERSION_ASK,
                "instructions": SYSTEM_INSTRUCTIONS,
                "scope": scope,
                "machine_id": machine_id,
                "lookback_days": lookback_days,
                "question_normalized": normalize_question(question),
                "facts": [project_assessment(snapshot) for snapshot in context],
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _tail_text() -> str:
    return "Non-claims: " + " ".join(NON_CLAIMS) + " " + FOOTER


def _base_output(
    context: list[Assessment],
    *,
    scope: str,
    question: str,
    lookback_days: int | None,
    window_start: datetime | None,
    window_end: datetime | None,
    history_truncated: bool,
    machine_id: str | None,
) -> dict[str, Any]:
    normalized = normalize_question(question)
    ids = [str(snapshot.assessment_id) for snapshot in context]
    return {
        "scope": scope,
        "assessment_id": ids[0] if scope == "assessment" else None,
        "machine_id": machine_id if scope == "machine" else None,
        "lookback_days": lookback_days,
        "window_start": window_start,
        "window_end": window_end,
        "history_truncated": history_truncated,
        "assessment_count": len(context),
        "question_sha256": question_digest(normalized),
        "question_echo": sanitize_echo(question),
        "present_modalities": [
            kind
            for kind in CANONICAL_KINDS
            if any(m.kind == kind and m.availability == "available" for s in context for m in s.modalities)
        ],
        "referenced_assessment_ids": ids,
        "referenced_evidence_ids": [str(item.evidence_id) for s in context for item in s.evidence][:250],
    }


def refusal_output_ask(
    context: list[Assessment],
    *,
    scope: str,
    question: str,
    lookback_days: int | None,
    window_start: datetime | None,
    window_end: datetime | None,
    history_truncated: bool,
    machine_id: str | None,
) -> AiAskOutput:
    """Deterministic refusal used when generation cannot proceed at all."""
    label = (
        f"machine {machine_id}" if scope == "machine"
        else f"assessment {context[0].assessment_id} for machine {context[0].machine_id}"
    )
    summary = "No answer could be produced from the retrieved stored evidence."
    detail = (
        "No supported answer could be produced from the retrieved stored evidence, so no answer "
        "is offered and nothing was fabricated."
    )
    text = TemplateAskProvider()._compose(
        label, "unsupported", [], summary, [detail],
        ["No interpretation is offered."], _limitations(context),
    )
    return AiAskOutput(
        feature="ask", intent="unsupported", matched_terms=[], answered=False,
        refusal_reason="evidence_insufficient", relevant_facts=[detail],
        uncertainty_indicators=["No interpretation is offered."], limitations=_limitations(context),
        non_claims=list(NON_CLAIMS), answer_summary=summary[:MAX_NOTE_CHARS],
        answer_text=text,
        **_base_output(
            context, scope=scope, question=question, lookback_days=lookback_days,
            window_start=window_start, window_end=window_end,
            history_truncated=history_truncated, machine_id=machine_id,
        ),
    )


def fallback_output_ask(
    context: list[Assessment],
    *,
    scope: str,
    question: str,
    lookback_days: int | None,
    window_start: datetime | None,
    window_end: datetime | None,
    history_truncated: bool,
    machine_id: str | None,
) -> AiAskOutput:
    """Deterministic fallback: generation or validation failed; say only stored scope."""
    label = (
        f"machine {machine_id}" if scope == "machine"
        else f"assessment {context[0].assessment_id} for machine {context[0].machine_id}"
    )
    summary = "A full answer could not be produced; only stored scope is reported."
    detail = (
        "A full answer could not be produced, so only the stored scope is reported and no "
        "interpretation is offered."
    )
    text = TemplateAskProvider()._compose(
        label, "unsupported", [], summary, [detail],
        ["The answer template did not validate; no interpretation is offered."], _limitations(context),
    )
    return AiAskOutput(
        feature="ask", intent="unsupported", matched_terms=[], answered=False,
        refusal_reason="evidence_insufficient", relevant_facts=[detail],
        uncertainty_indicators=["The answer template did not validate; no interpretation is offered."],
        limitations=_limitations(context), non_claims=list(NON_CLAIMS),
        answer_summary=summary[:MAX_NOTE_CHARS], answer_text=text,
        **_base_output(
            context, scope=scope, question=question, lookback_days=lookback_days,
            window_start=window_start, window_end=window_end,
            history_truncated=history_truncated, machine_id=machine_id,
        ),
    )


__all__ = [
    "ALL_KEYWORDS",
    "CANONICAL_KINDS",
    "FOOTER",
    "FORBIDDEN_PHRASES",
    "INTENT_KEYWORDS",
    "INTENT_ORDER",
    "NON_CLAIMS",
    "REQUIRED_PHRASES",
    "SUPPORTED_TOPICS",
    "TemplateAskProvider",
    "classify_question",
    "fallback_output_ask",
    "normalize_question",
    "question_digest",
    "refusal_output_ask",
    "sanitize_echo",
]
