"""Central AI Analysis Agent - slice D: Maintenance Report (deterministic, offline).

Builds a maintenance-oriented report from the immutable assessment snapshots that an
owner can see for ONE machine, inside a bounded lookback window resolved against stored
``created_at`` values only. It reuses the slice-A facts-only projection boundary
(``ai_template``), so it never calls a model, never reads raw audio/telemetry/paths and
never creates a score, p-value, confidence, status, id or timestamp.

The window is derived, never fabricated: ``window_end`` is the newest stored assessment
``created_at`` and ``window_start`` is ``lookback_days`` earlier. Wall-clock time is
never used as the anchor, so the same stored history always yields the same report.

Recommendations are NON-BINDING review suggestions derived only from stored patterns;
they never confirm a fault and never prescribe a maintenance action. Every report ends
with fixed non-claims: not a fault diagnosis, not machine-health completeness, not
coverage, not a fused failure probability/health score/risk score, not a guaranteed
maintenance plan.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from ..schemas.ai import (
    MAX_EXPLANATION_CHARS,
    MAX_REPORT_EVIDENCE,
    PROMPT_VERSION_REPORT,
    AiAvailabilityCount,
    AiModalityAvailability,
    AiRecurringFlag,
    AiReportOutput,
    AiReviewSuggestion,
    AiStatusCount,
)
from ..schemas.assessment import Assessment
from .ai_template import SYSTEM_INSTRUCTIONS, _clean, _iso, project_assessment, uncertaintynotes

CANONICAL_KINDS = ("sensor", "acoustic", "bearing")
CANONICAL_STATES = ("available", "unavailable", "unsupported", "invalid", "error")
CANONICAL_STATUSES = ("normal", "watch", "critical", "unknown")

# Fixed non-claims, identical on every report (structured field + text footer).
NON_CLAIMS = [
    "This report is not a fault diagnosis and does not confirm any fault.",
    "This report is not machine-health completeness and makes no coverage claim.",
    "This report is not coverage of machine health.",
    "This report is not a fused failure probability, health score or risk score.",
    "This report is not a guaranteed maintenance plan.",
]

FOOTER = (
    "This report summarizes stored assessment facts only. It is not a fault diagnosis, "
    "not machine-health completeness, and not coverage; it creates no fused failure "
    "probability, health score or risk score, and it is not a guaranteed maintenance plan."
)

REQUIRED_PHRASES = (
    "not a fault diagnosis",
    "not machine-health completeness",
    "not coverage",
    "not a fused failure probability",
    "not a guaranteed maintenance plan",
)

# Wording the agent must never emit: confirmed faults, guaranteed fixes, prescribed work.
FORBIDDEN_PHRASES = (
    "definitely faulty",
    "confirmed diagnosis",
    "will fix",
    "will prevent",
    "maintenance is required",
    "guaranteed to",
)


def status_counts(snapshots: list[Assessment]) -> dict[str, int]:
    """Stored overall-status counts over the retrieved snapshots (all four states)."""
    counts = {status: 0 for status in CANONICAL_STATUSES}
    for snapshot in snapshots:
        counts[snapshot.overall_status] += 1
    return counts


def status_distribution(snapshots: list[Assessment]) -> list[AiStatusCount]:
    """Stored overall-status counts, present statuses only (quoted values are stored)."""
    counts = status_counts(snapshots)
    return [
        AiStatusCount(status=status, count=counts[status])
        for status in CANONICAL_STATUSES
        if counts[status] > 0
    ]


def modality_availability(snapshots: list[Assessment]) -> list[AiModalityAvailability]:
    """Per-modality availability counts and distinct stored reason codes."""
    per_kind: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        for record in snapshot.modalities:
            row = per_kind.setdefault(record.kind, {"counts": {}, "reasons": set()})
            row["counts"][record.availability] = row["counts"].get(record.availability, 0) + 1
            row["reasons"].add(record.reason_code)
    rows: list[AiModalityAvailability] = []
    for kind in CANONICAL_KINDS:
        row = per_kind.get(kind)
        if row is None:
            continue
        rows.append(
            AiModalityAvailability(
                kind=kind,  # type: ignore[arg-type]
                counts=[
                    AiAvailabilityCount(state=state, count=row["counts"][state])
                    for state in CANONICAL_STATES
                    if row["counts"].get(state, 0) > 0
                ],
                reason_codes=sorted(row["reasons"]),
            )
        )
    return rows


def flag_counts(snapshots: list[Assessment]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for snapshot in snapshots:
        for flag in snapshot.quality_flags:
            key = (flag.code, flag.severity)
            counts[key] = counts.get(key, 0) + 1
    return counts


def recurring_flags(snapshots: list[Assessment]) -> list[AiRecurringFlag]:
    counts = flag_counts(snapshots)
    return [
        AiRecurringFlag(code=code, severity=severity, count=count)
        for (code, severity), count in sorted(counts.items())
    ]


def missing_data_patterns(
    snapshots: list[Assessment], *, history_truncated: bool
) -> list[str]:
    """Missing-data patterns derived only from stored availability and quality flags."""
    patterns: list[str] = []
    seen: set[str] = set()
    for snapshot in snapshots:
        for record in snapshot.modalities:
            if record.availability == "available":
                continue
            phrase = f'{record.kind} evidence is "{record.availability}" (stored reason "{record.reason_code}")'
            if phrase not in seen:
                seen.add(phrase)
                patterns.append(phrase)
    codes = {flag.code for snapshot in snapshots for flag in snapshot.quality_flags}
    if "TIMESTAMP_MISSING" in codes:
        patterns.append(
            '"TIMESTAMP_MISSING": a measurement timestamp is missing; no age or freshness '
            "judgement is made and none was fabricated"
        )
    if "MODEL_IDENTITY_INCOMPLETE" in codes:
        patterns.append(
            '"MODEL_IDENTITY_INCOMPLETE": model identity is incomplete; the missing metadata '
            "was not invented"
        )
    if history_truncated:
        patterns.append(
            "history was truncated to the newest assessments in the window; older in-window "
            "assessments were not analysed"
        )
    return patterns[:12]


def conservative_observations(snapshots: list[Assessment]) -> list[str]:
    return [
        "Stored statuses are conservative review priorities, never confirmed faults.",
        "No stored status, score or p-value is combined into a single score anywhere in this report.",
        "Co-occurrence of stored indications across assessments is described, not causation.",
    ]


def review_suggestions(
    snapshots: list[Assessment], *, history_truncated: bool
) -> list[AiReviewSuggestion]:
    """NON-BINDING review suggestions derived only from stored patterns."""
    counts = status_counts(snapshots)
    flags = flag_counts(snapshots)
    suggestions: list[AiReviewSuggestion] = []

    if counts.get("critical", 0) > 0:
        suggestions.append(
            AiReviewSuggestion(
                title="Review required: a stored review priority was recorded",
                basis=f'{counts["critical"]} assessment(s) stored an overall status of "critical"',
                detail=(
                    "One or more assessments stored a model-indicated review priority. This is a "
                    "non-binding review suggestion: manual verification of the machine may be "
                    "appropriate. It is not a confirmed fault, and no maintenance action is prescribed."
                ),
            )
        )
    if counts.get("watch", 0) > 0:
        suggestions.append(
            AiReviewSuggestion(
                title="Consider technician review: a stored watch status was recorded",
                basis=f'{counts["watch"]} assessment(s) stored an overall status of "watch"',
                detail=(
                    "One or more assessments stored a conservative watch status. This is a "
                    "non-binding review suggestion: consider technician review and continued "
                    "monitoring. It is not a confirmed fault, and no maintenance action is prescribed."
                ),
            )
        )
    if counts.get("unknown", 0) > 0:
        suggestions.append(
            AiReviewSuggestion(
                title="Review required: evidence was missing or uncertain",
                basis=f'{counts["unknown"]} assessment(s) stored an overall status of "unknown"',
                detail=(
                    "One or more assessments stored an unknown status because evidence was missing "
                    "or uncertain. This is a non-binding review suggestion: manual verification may "
                    "be appropriate. It is not a confirmed fault, and no maintenance action is prescribed."
                ),
            )
        )
    for (code, severity), count in sorted(flags.items()):
        if count < 2:
            continue
        suggestions.append(
            AiReviewSuggestion(
                title=f'Recurring stored flag "{code}"',
                basis=f'stored flag "{code}" recurred {count} times in the window',
                detail=(
                    "This stored quality flag recurred across assessments. This is a non-binding "
                    "review suggestion: review required to understand the repeated condition. It is "
                    "not a confirmed fault, and no maintenance action is prescribed."
                ),
            )
        )
    if any(record.availability != "available" for snapshot in snapshots for record in snapshot.modalities):
        suggestions.append(
            AiReviewSuggestion(
                title="Consider technician review: some requested evidence was missing",
                basis="one or more modality records in the window are not available",
                detail=(
                    "Some requested evidence did not produce an available result, which limits "
                    "interpretation. This is a non-binding review suggestion: manual verification may "
                    "be appropriate. It is not a confirmed fault, and no maintenance action is prescribed."
                ),
            )
        )
    if history_truncated:
        suggestions.append(
            AiReviewSuggestion(
                title="Review required: history was truncated",
                basis="more assessments were in the window than the report analysed",
                detail=(
                    "More assessments were inside the window than this report analysed, so the view "
                    "is partial. This is a non-binding review suggestion: consider a narrower window. "
                    "It is not a confirmed fault, and no maintenance action is prescribed."
                ),
            )
        )
    return suggestions[:10]


def report_limitations(
    snapshots: list[Assessment], *, history_truncated: bool
) -> list[str]:
    limits: list[str] = []
    seen: set[str] = set()
    for snapshot in snapshots:
        for record in snapshot.modalities:
            for item in record.limitations:
                flat = _clean(item, 600, quotes=False)
                if flat not in seen:
                    seen.add(flat)
                    limits.append(flat)
    limits.append(
        f"This report covers only the newest {len(snapshots)} owner-visible assessments within the "
        "requested window; it is not a complete machine history and not machine-health completeness."
    )
    if history_truncated:
        limits.append(
            "History was truncated: only the newest assessments in the window were analysed; older "
            "in-window assessments were not read."
        )
    limits.append(
        "Only stored assessments were read; no model was re-run and no telemetry or audio was accessed."
    )
    return limits[:50]


def report_uncertainty(
    snapshots: list[Assessment], *, history_truncated: bool
) -> list[str]:
    notes: list[str] = []
    if any(record.availability != "available" for snapshot in snapshots for record in snapshot.modalities):
        notes.append(
            "Some requested modalities did not produce available evidence; missing data limits "
            "interpretation and was not fabricated."
        )
    codes = {flag.code for snapshot in snapshots for flag in snapshot.quality_flags}
    if "TIMESTAMP_MISSING" in codes:
        notes.append("A measurement timestamp is missing in the window; no age or freshness judgement is made.")
    if "SOURCE_ILLUSTRATIVE" in codes:
        notes.append(
            "At least one assessment is illustrative demo/simulated data, not a real-machine measurement."
        )
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


class TemplateReportProvider:
    """Deterministic offline maintenance-report provider (no network, no ML, no clock)."""

    provider = "template"
    model: str | None = None

    def report(
        self,
        snapshots: list[Assessment],
        *,
        window_start: datetime | None,
        window_end: datetime | None,
        history_truncated: bool,
        lookback_days: int,
    ) -> AiReportOutput:
        machine_id = snapshots[0].machine_id
        distribution = status_distribution(snapshots)
        availability = modality_availability(snapshots)
        flags = recurring_flags(snapshots)
        patterns = missing_data_patterns(snapshots, history_truncated=history_truncated)
        observations = conservative_observations(snapshots)
        suggestions = review_suggestions(snapshots, history_truncated=history_truncated)
        limitations = report_limitations(snapshots, history_truncated=history_truncated)
        notes = report_uncertainty(snapshots, history_truncated=history_truncated)
        ids = [str(snapshot.assessment_id) for snapshot in snapshots]
        evidence_ids = [str(item.evidence_id) for snapshot in snapshots for item in snapshot.evidence]

        header = (
            f"Maintenance review report for machine {machine_id}. Window end {_iso(window_end)} (the "
            f"newest stored assessment created_at) and window start {_iso(window_start)} "
            f"({lookback_days} days earlier), both derived from stored created_at values."
        )
        count_line = (
            f"Assessments in the window (newest first): {len(snapshots)}. History truncated: "
            f'{"true" if history_truncated else "false"}.'
        )
        status_line = "Stored overall-status distribution: " + ", ".join(
            f'"{row.status}" {row.count}' for row in distribution
        )
        availability_lines = [
            f'{row.kind}: availability ' + ", ".join(f'"{c.state}" {c.count}' for c in row.counts)
            + f'; stored reason codes: {", ".join(row.reason_codes) or "none recorded"}.'
            for row in availability
        ]
        flags_line = (
            "Recurring stored quality flags: "
            + (
                "; ".join(f'"{flag.code}" ({flag.severity}) x{flag.count}' for flag in flags)
                if flags
                else "none stored."
            )
        )
        patterns_line = "Missing-data patterns: " + (
            "; ".join(patterns) if patterns else "none stored."
        )
        observations_line = "Conservative observations: " + " ".join(observations)
        suggestion_lines = [
            f"- review_suggestion [{suggestion.title}]: {suggestion.detail} (basis: {suggestion.basis})"
            for suggestion in suggestions
        ]
        suggestions_section = (
            "Non-binding review suggestions: " + " ".join(suggestion_lines)
            if suggestion_lines
            else "Non-binding review suggestions: none stored in this window."
        )
        limitations_line = "Limitations and uncertainty: " + (
            "; ".join(f"- {item}" for item in limitations) if limitations else "none stored"
        )
        uncertainty_line = "Uncertainty notes: " + (
            "; ".join(notes) if notes else "none stored."
        )

        head_sections = [
            header,
            count_line,
            status_line,
            *availability_lines,
            flags_line,
            patterns_line,
            observations_line,
            suggestions_section,
            limitations_line,
            uncertainty_line,
        ]
        # Guarantee the required non-claim language survives length bounding.
        tail = "Non-claims: " + " ".join(NON_CLAIMS) + " " + FOOTER
        head = _clean(" ".join(head_sections))
        tail = _clean(tail)
        budget = max(0, MAX_EXPLANATION_CHARS - len(tail) - 1)
        report_text = (head[:budget].rstrip() + " " + tail)[:MAX_EXPLANATION_CHARS]

        return AiReportOutput(
            feature="report",
            machine_id=machine_id,
            window_start=window_start,
            window_end=window_end,
            assessment_count=len(snapshots),
            history_truncated=history_truncated,
            referenced_assessment_ids=ids,
            modality_availability=availability,
            status_distribution=distribution,
            recurring_flags=flags,
            missing_data_patterns=patterns,
            conservative_observations=observations,
            review_suggestions=suggestions,
            limitations=limitations,
            uncertainty_notes=notes,
            non_claims=list(NON_CLAIMS),
            report_text=report_text,
            referenced_evidence_ids=evidence_ids[:MAX_REPORT_EVIDENCE],
        )

    def prompt_sha256(
        self,
        machine_id: str,
        snapshots: list[Assessment],
        *,
        window_start: datetime | None,
        window_end: datetime | None,
        history_truncated: bool,
        lookback_days: int,
    ) -> str:
        """Deterministic digest of the effective prompt (version + instructions + facts)."""
        canonical = json.dumps(
            {
                "prompt_version": PROMPT_VERSION_REPORT,
                "instructions": SYSTEM_INSTRUCTIONS,
                "machine_id": machine_id,
                "lookback_days": lookback_days,
                "window_start": _iso(window_start),
                "window_end": _iso(window_end),
                "history_truncated": history_truncated,
                "facts": [project_assessment(snapshot) for snapshot in snapshots],
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _tail_text() -> str:
    return "Non-claims: " + " ".join(NON_CLAIMS) + " " + FOOTER


def refusal_output_report(machine_id: str, *, lookback_days: int) -> AiReportOutput:
    """Deterministic refusal: no owner-visible assessments inside the window."""
    text = _clean(
        f"No owner-visible assessments were found for machine {machine_id} within the requested "
        f"{lookback_days}-day window, so no maintenance report is offered. The window is resolved "
        "against stored assessment timestamps only, and no facts were fabricated to fill it. "
        + _tail_text()
    )
    return AiReportOutput(
        feature="report",
        machine_id=machine_id,
        window_start=None,
        window_end=None,
        assessment_count=0,
        history_truncated=False,
        referenced_assessment_ids=[],
        modality_availability=[],
        status_distribution=[],
        recurring_flags=[],
        missing_data_patterns=[],
        conservative_observations=[],
        review_suggestions=[],
        limitations=[],
        uncertainty_notes=["No owner-visible assessment was stored for this machine inside the window."],
        non_claims=list(NON_CLAIMS),
        report_text=text,
        referenced_evidence_ids=[],
    )


def fallback_output_report(
    snapshots: list[Assessment],
    *,
    machine_id: str,
    window_start: datetime | None,
    window_end: datetime | None,
    history_truncated: bool,
) -> AiReportOutput:
    """Deterministic fallback: generation or validation failed; report stored scope only."""
    ids = [str(snapshot.assessment_id) for snapshot in snapshots]
    text = _clean(
        f"A maintenance review report covering {len(snapshots)} assessment(s) could not be fully "
        f"produced for machine {machine_id}, so only the stored window and counts are reported and "
        "no interpretation is offered. " + _tail_text()
    )
    return AiReportOutput(
        feature="report",
        machine_id=machine_id,
        window_start=window_start,
        window_end=window_end,
        assessment_count=len(snapshots),
        history_truncated=history_truncated,
        referenced_assessment_ids=ids,
        modality_availability=[],
        status_distribution=[],
        recurring_flags=[],
        missing_data_patterns=[],
        conservative_observations=[],
        review_suggestions=[],
        limitations=[],
        uncertainty_notes=["The report template did not validate; no interpretation is offered."],
        non_claims=list(NON_CLAIMS),
        report_text=text,
        referenced_evidence_ids=[],
    )


__all__ = [
    "CANONICAL_KINDS",
    "CANONICAL_STATES",
    "CANONICAL_STATUSES",
    "NON_CLAIMS",
    "FOOTER",
    "REQUIRED_PHRASES",
    "FORBIDDEN_PHRASES",
    "TemplateReportProvider",
    "fallback_output_report",
    "flag_counts",
    "missing_data_patterns",
    "modality_availability",
    "recurring_flags",
    "refusal_output_report",
    "report_limitations",
    "status_counts",
    "status_distribution",
]
