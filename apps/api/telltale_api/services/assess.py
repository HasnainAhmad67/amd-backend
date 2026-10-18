"""Pure assessment assembly + quality/status rules (policy ``assess/v1``).

No HTTP, no inference orchestration, no LLM: this module only imports
``schemas.assessment`` (pydantic) so it runs standalone.

Status policy assess/v1 (exact rule table):

Modality indications (available evidence only; non-available => None):
  sensor    risk low/medium/high  -> normal/watch/critical
  acoustic  status normal/watch/anomalous -> normal/watch/critical
  bearing   conservative mapping, NEVER critical:
      classifier uncertain flag set        -> unknown  (bearing:classifier_uncertain)
      prediction_set empty                 -> unknown  (bearing:empty_prediction_set)
      no class probabilities               -> unknown  (bearing:no_class_probabilities)
      predicted_class not in set           -> unknown  (bearing:inconsistent_output)
      set contains only "normal"           -> normal
      pred == "normal" but set has faults  -> unknown  (bearing:ambiguous_normal_with_faults)
      set has faults, pred is a fault      -> watch    (bearing:fault_classes_in_set)
      A set containing fault classes alone is NOT a confirmed fault.

"critical" = model-indicated review priority. It is NOT a confirmed mechanical
fault, a certified danger, or autonomous control advice.

Overall status (max-rank over available modalities only; never a fused average):
  1. any critical indication            -> critical (may coexist with incomplete evidence)
  2. else any watch                     -> watch    (may coexist with incomplete evidence)
  3. no available evidence              -> unknown  (+ NO_AVAILABLE_MODALITY)
  4. else any available-but-unknown     -> unknown  (uncertain evidence precludes "normal")
  5. else present < expected            -> unknown  (missing required evidence is NEVER normal;
                                                     individual normal indications stay preserved)
  6. else                               -> normal

Completeness = present/expected over the preserved expected-modality list;
it measures evidence presence, NOT confidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence
from uuid import uuid4

from ..schemas.assessment import (
    SCHEMA_VERSION,
    Assessment,
    AvailabilityState,
    EvidenceCompleteness,
    EvidenceDraft,
    EvidenceItem,
    ModalityDraft,
    ModalityKind,
    ModalityRecord,
    OverallStatus,
    QualityFlag,
    SourceMode,
)
from ..schemas.acoustic import FaultDiagnosis

RULE_VERSION = "assess/v1"

_SENSOR_MAP = {"low": "normal", "medium": "watch", "high": "critical"}
_ACOUSTIC_MAP = {"normal": "normal", "watch": "watch", "anomalous": "critical"}


def _server_now() -> datetime:
    return datetime.now(timezone.utc)


def bearing_indication(output: FaultDiagnosis) -> tuple[OverallStatus, list[str]]:
    """Conservative bearing mapping. Never returns ``critical``."""
    if output.uncertain:
        return "unknown", ["bearing:classifier_uncertain"]
    prediction_set = list(dict.fromkeys(output.prediction_set))  # de-dup, keep order
    if not prediction_set:
        return "unknown", ["bearing:empty_prediction_set"]
    if not output.probabilities:
        return "unknown", ["bearing:no_class_probabilities"]
    predicted = output.predicted_class.strip()
    if predicted not in prediction_set:
        return "unknown", ["bearing:inconsistent_output"]
    fault_classes = [c for c in prediction_set if c != "normal"]
    if not fault_classes:
        return "normal", []
    if predicted == "normal":
        # Fault classes are in the set although "normal" was predicted: ambiguous,
        # so we do NOT silently label the machine healthy.
        return "unknown", ["bearing:ambiguous_normal_with_faults"]
    # Fault classes in the set with a fault prediction: review priority only.
    return "watch", ["bearing:fault_classes_in_set"]


def modality_indication(record: ModalityRecord) -> tuple[OverallStatus | None, list[str]]:
    """Per-modality indication. ``None`` = no usable evidence (modality not available)."""
    if record.availability != "available" or record.output is None:
        return None, []
    if record.kind == "sensor":
        return _SENSOR_MAP[record.output.risk_level], []  # type: ignore[attr-defined]
    if record.kind == "acoustic":
        return _ACOUSTIC_MAP[record.output.status], []  # type: ignore[attr-defined]
    if record.kind == "bearing":
        return bearing_indication(record.output)  # type: ignore[arg-type]
    raise ValueError(f"unknown modality kind: {record.kind}")


def _identity_incomplete(record: ModalityRecord) -> bool:
    if record.availability != "available":
        return False
    identity = record.model_identity
    if identity is None:
        return True
    return any(
        getattr(identity, field) is None
        for field in ("version", "artifact_fingerprint", "preprocessing_fingerprint")
    )


def _evaluate(
    *,
    source_mode: SourceMode,
    expected: Sequence[ModalityKind],
    records: Sequence[ModalityRecord],
    evidence: Sequence[EvidenceItem],
) -> tuple[OverallStatus, list[str], list[QualityFlag]]:
    """Returns (overall_status, status_reasons, quality_flags) - deterministic, no LLM."""
    expected_n = len(expected)
    present_n = sum(1 for r in records if r.availability == "available")
    reasons: list[str] = []
    flags: list[QualityFlag] = []

    # Availability reasons (one explicit entry per non-available expected modality).
    by_kind = {r.kind: r for r in records}
    for kind in expected:
        record = by_kind[kind]
        if record.availability != "available":
            reasons.append(f"{kind}:{record.availability}:{record.reason_code}")

    # Per-modality indications. Uncertainty reasons (from the indication rules,
    # e.g. bearing ambiguity) are tracked separately from availability reasons so
    # the BEARING_UNCERTAIN flag never fires for a merely-missing modality.
    indications: dict[str, OverallStatus] = {}
    uncertainty_reasons: list[str] = []
    for kind in expected:
        indication, extra = modality_indication(by_kind[kind])
        if indication is not None:
            indications[kind] = indication
            reasons.extend(extra)
            uncertainty_reasons.extend(extra)

    # --- quality flags (deduplicated, templated explanations only) ---
    if present_n == 0:
        flags.append(
            QualityFlag(
                code="NO_AVAILABLE_MODALITY",
                severity="warning",
                explanation="No expected modality produced available evidence; overall status is "
                "unknown (missing evidence never implies normal).",
            )
        )
    if present_n < expected_n:
        flags.append(
            QualityFlag(
                code="EVIDENCE_INCOMPLETE",
                severity="warning",
                explanation=f"Only {present_n}/{expected_n} expected modalities produced available "
                "evidence; completeness measures presence, not confidence.",
            )
        )

    all_source_modes = [source_mode] + [r.source_mode for r in records]
    all_source_modes += [e.source_mode for e in evidence]
    illustrative = any(m in ("dataset_demo", "simulated") for m in all_source_modes)
    if illustrative:
        flags.append(
            QualityFlag(
                code="SOURCE_ILLUSTRATIVE",
                severity="info",
                explanation="One or more inputs are dataset_demo/simulated: illustrative demo "
                "data, not real-machine measurements.",
            )
        )

    available_modes = {r.source_mode for r in records if r.availability == "available"}
    if len(available_modes) > 1:
        flags.append(
            QualityFlag(
                code="MIXED_SOURCE_PROVENANCE",
                severity="warning",
                explanation="Available evidence combines modalities from different source modes; "
                "the sources are unrelated (interpret the combined status with care).",
            )
        )

    missing_timestamps = any(r.observed_at is None for r in records if r.availability == "available")
    missing_timestamps = missing_timestamps or any(e.observed_at is None for e in evidence)
    if missing_timestamps:
        flags.append(
            QualityFlag(
                code="TIMESTAMP_MISSING",
                severity="warning",
                explanation="A supplied item has no measurement timestamp; none was fabricated "
                "(age checking stays disabled until a freshness policy is approved).",
            )
        )

    if any(_identity_incomplete(r) for r in records):
        flags.append(
            QualityFlag(
                code="MODEL_IDENTITY_INCOMPLETE",
                severity="warning",
                explanation="Model identity is incomplete (missing artifact/preprocessing "
                "identity); the missing metadata was not invented.",
            )
        )

    if uncertainty_reasons:
        flags.append(
            QualityFlag(
                code="BEARING_UNCERTAIN",
                severity="warning",
                explanation="Bearing output is ambiguous/empty/inconsistent; a conservative "
                "review status is reported, not a confirmed fault.",
            )
        )

    if any(r.limitations for r in records):
        flags.append(
            QualityFlag(
                code="MODEL_LIMITATIONS_PRESENT",
                severity="info",
                explanation="Model applicability/generalization limitations are attached to this "
                "evidence; no accuracy guarantee is implied.",
            )
        )

    # --- overall status (max rank; never averages unlike scores) ---
    values = list(indications.values())
    if any(v == "critical" for v in values):
        overall: OverallStatus = "critical"
    elif any(v == "watch" for v in values):
        overall = "watch"
    elif present_n == 0:
        overall = "unknown"
        reasons.append("no_usable_evidence")
    elif any(v == "unknown" for v in values):
        overall = "unknown"
        reasons.append("uncertain_evidence_precludes_normal")
    elif present_n < expected_n:
        overall = "unknown"
        reasons.append("incomplete_evidence_precludes_normal")
    else:
        overall = "normal"

    return overall, reasons, flags


def create_assessment(
    *,
    machine_id: str,
    source_mode: SourceMode,
    expected_modalities: Sequence[ModalityKind],
    modalities: Sequence[ModalityDraft] = (),
    evidence: Sequence[EvidenceDraft] = (),
    created_at: datetime | None = None,
    extra_flags: Sequence[QualityFlag] = (),
) -> Assessment:
    """Assemble an immutable assessment. ALL authoritative fields are server-owned:

    assessment_id / created_at (UTC now unless a fixed clock is injected for tests),
    schema_version, rule_version, per-modality result_id, evidence ids, illustrative
    flags, quality flags, status reasons, completeness and overall status.

    ``extra_flags`` lets a route attach DETERMINISTIC, already-typed quality facts it
    observed at the boundary (e.g. AUDIO_TRUNCATED for an upload that exceeded the
    model's processing cap). Route-supplied flags never change status computation;
    they are appended after the computed flags and must use existing QualityCode
    values - routes cannot invent authoritative statuses through them.
    """
    expected = list(expected_modalities)
    if not expected:
        raise ValueError("expected_modalities must not be empty")

    supplied = {m.kind for m in modalities}
    # Expected-but-absent modalities become explicit `unavailable/not_submitted`
    # placeholders, so completeness is deterministic.
    records: list[ModalityRecord] = list(modalities)
    for kind in expected:
        if kind not in supplied:
            records.append(
                ModalityDraft(
                    kind=kind,
                    availability="unavailable",  # type: ignore[arg-type]
                    reason_code="not_submitted",  # type: ignore[arg-type]
                    source_mode=source_mode,
                )
            )

    final_records: list[ModalityRecord] = []
    for record in records:
        illustrative = record.source_mode in ("dataset_demo", "simulated")
        final_records.append(
            ModalityRecord.model_validate(
                {
                    **record.model_dump(),
                    "result_id": uuid4(),
                    "illustrative": illustrative,
                }
            )
        )

    final_evidence: list[EvidenceItem] = []
    for item in evidence:
        illustrative = item.source_mode in ("dataset_demo", "simulated")
        final_evidence.append(
            EvidenceItem.model_validate(
                {**item.model_dump(), "evidence_id": uuid4(), "illustrative": illustrative}
            )
        )

    overall, reasons, flags = _evaluate(
        source_mode=source_mode,
        expected=expected,
        records=final_records,
        evidence=final_evidence,
    )
    flags.extend(extra_flags)  # deterministic boundary facts (e.g. truncation), after computed flags
    present_n = sum(1 for r in final_records if r.availability == "available")

    return Assessment(
        assessment_id=uuid4(),
        machine_id=machine_id,
        source_mode=source_mode,
        schema_version=SCHEMA_VERSION,
        rule_version=RULE_VERSION,
        created_at=created_at or _server_now(),
        expected_modalities=expected,  # type: ignore[arg-type]
        modalities=final_records,
        evidence=final_evidence,
        quality_flags=flags,
        overall_status=overall,
        status_reasons=reasons,
        evidence_completeness=EvidenceCompleteness(
            expected=len(expected),
            present=present_n,
            ratio=present_n / len(expected),
        ),
        approved_playbook_item_ids=[],
    )


__all__ = [
    "RULE_VERSION",
    "AvailabilityState",
    "bearing_indication",
    "create_assessment",
    "modality_indication",
]
