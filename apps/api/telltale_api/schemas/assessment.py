"""Immutable assessment contracts (foundation slice: schemas only - no endpoints).

Server-owned fields (assessment_id, created_at, schema_version, rule_version, model
outputs, evidence ids, result ids) are stamped by ``services.assess.create_assessment``;
every model here uses ``extra="forbid"`` so future clients cannot smuggle authoritative
scores, statuses, versions or ids through these documents.

Notes:
- ``evidence_completeness`` measures how many EXPECTED modalities produced available
  evidence. It is NOT confidence and must never be read as such.
- ``source_mode`` (provenance) is structurally separate from
  ``model_identity.training_dataset``: an uploaded recording stays "uploaded" even when
  the scoring model was trained on MIMII/demo data.
- Timestamps must be timezone-aware when supplied and are normalized to UTC. A missing
  measurement timestamp stays ``None`` (never fabricated from receipt time) and is
  surfaced by a quality flag. There are deliberately NO freshness/future-skew limits
  until a policy is separately approved.
- ``approved_playbook_item_ids`` is reserved and always empty in this slice: no playbook
  mapping or generated inspection guidance exists yet.
- Immutability provided by the store is application-level (no update/replace/delete
  API), not tamper-proof storage against a database administrator.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .acoustic import AcousticAnalysis, FaultDiagnosis
from .sensor import SensorAssessment, SensorReading

SCHEMA_VERSION = "1.0"

SourceMode = Literal["dataset_demo", "simulated", "uploaded", "live"]
ModalityKind = Literal["sensor", "acoustic", "bearing"]
AvailabilityState = Literal["available", "unavailable", "unsupported", "invalid", "error"]
# Explicit reason codes: omitted modality / unloaded model / unsupported input /
# invalid input / inference error are never conflated.
AvailabilityReason = Literal[
    "ok", "not_submitted", "model_not_loaded", "unsupported_input", "invalid_input", "inference_error"
]
REASON_BY_STATE: dict[str, tuple[str, ...]] = {
    "available": ("ok",),
    "unavailable": ("not_submitted", "model_not_loaded"),
    "unsupported": ("unsupported_input",),
    "invalid": ("invalid_input",),
    "error": ("inference_error",),
}

OverallStatus = Literal["normal", "watch", "critical", "unknown"]
QualitySeverity = Literal["info", "warning", "critical"]
QualityCode = Literal[
    "EVIDENCE_INCOMPLETE",
    "NO_AVAILABLE_MODALITY",
    "SOURCE_ILLUSTRATIVE",
    "MIXED_SOURCE_PROVENANCE",
    "TIMESTAMP_MISSING",
    "MODEL_IDENTITY_INCOMPLETE",
    "BEARING_UNCERTAIN",
    "MODEL_LIMITATIONS_PRESENT",
    "AUDIO_TRUNCATED",  # upload exceeded the model's processing cap; only part of it was scored
]

DEMO_SOURCE_MODES = ("dataset_demo", "simulated")

_OUTPUT_TYPE: dict[str, type] = {
    "sensor": SensorAssessment,
    "acoustic": AcousticAnalysis,
    "bearing": FaultDiagnosis,
}


def _to_aware_utc(value: datetime | None) -> datetime | None:
    """Reject naive/malformed timestamps; normalize aware ones to UTC. No skew limits."""
    if value is None:
        return None
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware (naive timestamps are rejected)")
    return value.astimezone(timezone.utc)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelIdentity(_Strict):
    """Artifact + preprocessing identity of the model that produced a modality output.

    Missing identity is represented explicitly (None) and flagged by the quality rules;
    it is never invented. ``version`` alone (e.g. "0.1.0") is NOT sufficient identity.
    """

    name: str | None = None
    version: str | None = None
    trained_at: datetime | None = None
    artifact_fingerprint: str | None = None  # sha256 of the loaded artifact file(s)
    preprocessing_fingerprint: str | None = None  # None => unknown preprocessing identity
    training_dataset: str | None = None  # what the MODEL was trained on (not provenance)

    @field_validator("trained_at")
    @classmethod
    def _norm_trained(cls, value: datetime | None) -> datetime | None:
        return _to_aware_utc(value)


class EvidenceDraft(_Strict):
    """Author-supplied (server-side) evidence input; ids/flags are server-owned."""

    kind: str
    value: float | int | str | bool
    unit: str | None = None
    source: str
    source_mode: SourceMode
    observed_at: datetime | None = None

    @field_validator("observed_at")
    @classmethod
    def _norm_observed(cls, value: datetime | None) -> datetime | None:
        return _to_aware_utc(value)


class EvidenceItem(EvidenceDraft):
    evidence_id: UUID  # server-generated
    illustrative: bool = False  # server-computed (dataset_demo/simulated inputs)


class ModalityDraft(_Strict):
    """Server-authored per-modality snapshot input; result_id/illustrative are server-owned."""

    kind: ModalityKind
    availability: AvailabilityState
    reason_code: AvailabilityReason
    source_mode: SourceMode
    observed_at: datetime | None = None  # measurement time; None = unknown (never fabricated)
    inference_at: datetime | None = None  # server inference time (distinct from observed_at)
    model_identity: ModelIdentity | None = None
    recording_id: str | None = None
    reference_id: str | None = None
    limitations: list[str] = []  # model applicability/generalization caveats, kept as-is
    output: SensorAssessment | AcousticAnalysis | FaultDiagnosis | None = None
    error_detail: str | None = None
    # Hash-only retention: the sha256 of the raw uploaded bytes (lowercase hex).
    # Identity/integrity record of the upload - NOT replay capability (the audio
    # itself is never stored). None for modalities without an uploaded payload.
    recording_sha256: str | None = None

    @field_validator("recording_sha256")
    @classmethod
    def _check_recording_sha(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("recording_sha256 must be a lowercase hex SHA-256 digest")
        return value

    @field_validator("observed_at", "inference_at")
    @classmethod
    def _norm_timestamps(cls, value: datetime | None) -> datetime | None:
        return _to_aware_utc(value)

    @model_validator(mode="before")
    @classmethod
    def _coerce_output_by_kind(cls, data):
        # JSON reload path: validate the output against the type matching its modality kind.
        if isinstance(data, dict) and isinstance(data.get("output"), dict):
            target = _OUTPUT_TYPE.get(data.get("kind"))
            if target is not None:
                data = dict(data)
                data["output"] = target.model_validate(data["output"])
        return data

    @model_validator(mode="after")
    def _state_reason_output_consistent(self):
        allowed = REASON_BY_STATE[self.availability]
        if self.reason_code not in allowed:
            raise ValueError(
                f"reason_code '{self.reason_code}' does not match availability '{self.availability}' "
                f"(allowed: {', '.join(allowed)})"
            )
        if self.availability == "available":
            if self.output is None:
                raise ValueError("availability 'available' requires a modality output")
            if self.error_detail is not None:
                raise ValueError("availability 'available' cannot carry error_detail")
        else:
            if self.output is not None:
                raise ValueError(f"availability '{self.availability}' cannot carry a modality output")
        if self.output is not None and not isinstance(self.output, _OUTPUT_TYPE[self.kind]):
            raise ValueError(f"output type does not match modality kind '{self.kind}'")
        return self


class ModalityRecord(ModalityDraft):
    result_id: UUID  # server-generated (full UUID)
    illustrative: bool = False  # server-computed


class EvidenceCompleteness(_Strict):
    """present/expected over the PRESERVED expected-modality list. Completeness is not confidence."""

    expected: int = Field(ge=1)
    present: int = Field(ge=0)
    ratio: float = Field(ge=0.0, le=1.0)


class QualityFlag(_Strict):
    code: QualityCode
    severity: QualitySeverity
    explanation: str


class Assessment(_Strict):
    """Immutable assessment snapshot (application-level immutability via the store)."""

    assessment_id: UUID
    machine_id: str = Field(min_length=1)
    source_mode: SourceMode  # assessment-level provenance (never model.dataset)
    schema_version: str
    rule_version: str
    created_at: datetime  # server-generated UTC, always
    expected_modalities: list[ModalityKind] = Field(min_length=1)
    modalities: list[ModalityRecord] = Field(min_length=1)
    evidence: list[EvidenceItem]
    quality_flags: list[QualityFlag]
    overall_status: OverallStatus
    status_reasons: list[str]
    evidence_completeness: EvidenceCompleteness
    approved_playbook_item_ids: list[str] = []  # reserved: empty (no playbook mapping yet)

    @field_validator("created_at")
    @classmethod
    def _norm_created(cls, value: datetime) -> datetime:
        return _to_aware_utc(value)  # type: ignore[return-value]

    @model_validator(mode="after")
    def _expected_matches_modalities(self):
        if len(set(self.expected_modalities)) != len(self.expected_modalities):
            raise ValueError("expected_modalities contains duplicates")
        kinds = [m.kind for m in self.modalities]
        if len(set(kinds)) != len(kinds):
            raise ValueError("duplicate modality records for the same kind")
        if set(kinds) != set(self.expected_modalities):
            raise ValueError(
                "modalities must cover expected_modalities exactly "
                f"(expected={sorted(self.expected_modalities)}, present={sorted(kinds)})"
            )
        return self


# ------------------------------------------------------------------ request contracts


DemoSourceMode = Literal["dataset_demo", "simulated"]


class StrictSensorReading(SensorReading):
    """Strict boundary for assessment requests.

    Same fields, bounds and defaults as ``SensorReading`` (legacy /sensors routes
    keep their existing behaviour untouched), but forged nested fields are
    REJECTED here instead of silently ignored.
    """

    model_config = ConfigDict(extra="forbid")


class AssessmentCreateRequest(_Strict):
    """Sensor-only assessment request (slice 1).

    Only inputs and provenance are accepted. Scores, statuses, versions, result
    ids, assessment ids, owner and expected_modalities are server-owned and can
    never be supplied by a client (rejected via ``extra="forbid"``).
    """

    reading: StrictSensorReading
    machine_id: str = Field(min_length=1)
    source_mode: DemoSourceMode
    observed_at: datetime | None = None  # measurement time; None = unknown (never fabricated)

    @field_validator("observed_at")
    @classmethod
    def _norm_observed(cls, value: datetime | None) -> datetime | None:
        return _to_aware_utc(value)


class SensorSubmission(_Strict):
    """Sensor input carried in the multimodal multipart field ``sensor_json``.

    Client inputs only: the reading plus its demo provenance - exactly the legacy
    sensor request's inputs minus ``machine_id``/``observed_at``, which are
    top-level multipart fields of that contract. Scores, statuses, ids,
    references, model identities, owner and expected_modalities stay
    server-owned and are rejected via ``extra="forbid"`` (forged nested reading
    fields are rejected too, not ignored). ``source_mode`` accepts only the
    legacy demo values, so a client can never forge assessment-level
    ``uploaded``/``live`` provenance through this field.
    """

    reading: StrictSensorReading
    source_mode: DemoSourceMode


class AssessmentListResponse(_Strict):
    """Owner-scoped, bounded listing envelope."""

    items: list[Assessment]
    limit: int
    offset: int
