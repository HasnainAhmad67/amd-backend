"""Central AI Analysis Agent routes - slice A: AI Result Explanation.

Two endpoints only (additive; existing assessment routes are untouched):

- ``POST /api/v1/ai/explain`` - read ONE immutable assessment snapshot through the
  owner-scoped assessment store, render a deterministic template explanation from
  its stored facts, ground-validate it, and append the validated output to the
  append-only AI output store.
- ``GET /api/v1/ai/outputs/{ai_output_id}`` - owner-scoped read back of one
  stored AI output. No list/search endpoints in this slice.

Safety rules enforced here:
- bearer auth + owner resolution BEFORE any evidence is read or any AI work runs;
- the AI agent never calls ML models, never reads raw audio or fleet telemetry, and
  never modifies the snapshot it explains (read-only use of the assessment store);
- cross-owner and unknown ids share one 404 response (no existence leakage);
- an output that fails grounding validation is never persisted as ``validated``;
- every failure path persists nothing and leaves stored assessments unchanged.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from ...schemas.ai import (
    MAX_IDS,
    PROMPT_VERSION,
    PROMPT_VERSION_EXPLAIN,
    PROMPT_VERSION_REVIEW,
    PROMPT_VERSION_ANALYZE,
    PROMPT_VERSION_REPORT,
    PROMPT_VERSION_ASK,
    AiAnalyzeOutput,
    AiAnalyzeRequest,
    AiAnalyzeResponse,
    AiAskOutput,
    AiAskRequest,
    AiAskResponse,
    AiEvidenceReviewOutput,
    AiEvidenceReviewRequest,
    AiEvidenceReviewResponse,
    AiExplainOutput,
    AiExplainRequest,
    AiExplainResponse,
    AiOutputRecord,
    AiReportOutput,
    AiReportRequest,
    AiReportResponse,
)
from ...services.ai_analyze_template import (
    TemplateAnalyzeProvider,
    fallback_output_analyze,
    refusal_output_analyze,
)
from ...services.ai_analyze_validate import validate_analyze_output
from ...services.ai_ask_template import (
    fallback_output_ask,
    normalize_question,
    question_digest,
    refusal_output_ask,
    sanitize_echo,
)
from ...services.ai_ask_validate import validate_ask_output
from ...services.ai_output_store import AiOutputStore
from ...services.ai_provider import get_ai_provider
from ...services.ai_report_template import (
    fallback_output_report,
    refusal_output_report,
)
from ...services.ai_report_validate import validate_report_output
from ...services.ai_review_template import (
    TemplateEvidenceReviewProvider,
    fallback_output_review,
    refusal_output_review,
)
from ...services.ai_review_validate import validate_review_output
from ...services.ai_template import TemplateExplainProvider, fallback_output, refusal_output
from ...services.ai_validate import validate_explain_output
from ..deps import AppState, get_state

# Shared route-package helpers (auth + the canonical error envelope + validation
# detail formatting) defined once in the assessment routes module.
from .assessments import _authenticate, _error, _validation_details

log = logging.getLogger("telltale.ai")

router = APIRouter()

MAX_BODY_BYTES = 65536  # this endpoint's request is a single UUID-sized JSON object

# Newest-first retrieval cap for the maintenance report (MAX_IDS == 50). One extra row
# is fetched so truncation inside the window can be disclosed truthfully.
REPORT_MAX_ASSESSMENTS = MAX_IDS

# Ask Telltale (slice E) uses the same newest-first, owner-scoped, capped retrieval.
ASK_MAX_ASSESSMENTS = MAX_IDS


def _provider_failure_label(exc: BaseException) -> str:
    """A bounded, sanitized failure label - never a raw provider error body."""
    failure_class = getattr(exc, "failure_class", None)
    if failure_class is not None:
        return str(getattr(failure_class, "value", failure_class))
    return type(exc).__name__


def _assemble_amd_explain(snapshot: Any, prose: Any) -> AiExplainOutput:
    """Backend-owned structural fields + the AMD prose.

    Every structural field, id, status, flag code, reference and limitation comes from
    the deterministic template explanation of the immutable snapshot; AMD only supplies
    ``explanation_text`` (and optional uncertainty notes). Provider-invented ids,
    statuses or scores are therefore never trusted - and anything ungrounded in the
    prose is rejected by the unchanged grounding validator before persistence.
    """
    baseline = TemplateExplainProvider().explain(snapshot)
    notes = [str(note) for note in (getattr(prose, "uncertainty_notes", None) or [])][:12]
    return AiExplainOutput.model_validate(
        {
            **baseline.model_dump(),
            "explanation_text": getattr(prose, "explanation_text"),
            "uncertainty_notes": notes or baseline.uncertainty_notes,
        }
    )


def _explain_with_provider(
    provider: Any,
    template_provider: Any,
    request: Any,
    snapshot: Any,
    request_id: str,
) -> tuple[AiExplainOutput, str, str, str | None, str] | None:
    """Run one explain generation and ground-validate it.

    Returns ``(output, validation_status, provider_name, provider_model, prompt_digest)``
    or ``None`` when even the deterministic fallback could not be grounded (a bug that
    must surface as a server error, never as an invalid persisted record).
    """
    provider_name = "template"
    provider_model: str | None = None
    prompt_digest = template_provider.prompt_sha256(request)
    try:
        result = provider.generate(request)
        is_amd = getattr(result, "provider", "template") == "amd"
        output = (
            _assemble_amd_explain(snapshot, result.candidate) if is_amd else result.candidate
        )
        ok, reason = validate_explain_output(output, snapshot)
    except Exception as exc:
        log.info(
            "AI explain generation failed (%s, request_id=%s); using fallback",
            _provider_failure_label(exc),
            request_id,
        )
        output = fallback_output(snapshot)
        validation_status = "fallback"
    else:
        if ok:
            validation_status = "validated"
            if is_amd:
                # Provenance reflects the ACTUAL generator: an accepted AMD candidate.
                provider_name = "amd"
                provider_model = provider.model
                prompt_digest = provider.prompt_sha256(request)
        else:
            log.info("AI output failed grounding validation (%s); using fallback", reason)
            output = fallback_output(snapshot)
            validation_status = "fallback"

    if validation_status == "fallback":
        grounded, _ = validate_explain_output(output, snapshot)
        if not grounded:
            output = refusal_output(snapshot)
            validation_status = "refused"
            grounded, _ = validate_explain_output(output, snapshot)
            if not grounded:  # pragma: no cover - a bug would be a server error
                return None
    return output, validation_status, provider_name, provider_model, prompt_digest


def _assemble_amd_analyze(snapshot: Any, prose: Any) -> AiAnalyzeOutput:
    """Backend-owned structural fields + the AMD co-occurrence prose.

    Every structural field, id, status, modality summary, pairwise relation, agreement
    value, provenance line, reference and non-claim comes from the deterministic template
    analysis of the immutable snapshot; AMD only supplies ``analysis_text``. Provider-
    invented ids, statuses, relations or scores are therefore never trusted - and anything
    ungrounded in the prose is rejected by the unchanged grounding validator.
    """
    baseline = TemplateAnalyzeProvider().analyze(snapshot)
    return AiAnalyzeOutput.model_validate(
        {**baseline.model_dump(), "analysis_text": getattr(prose, "analysis_text")}
    )


def _analyze_with_provider(
    provider: Any,
    template_provider: Any,
    request: Any,
    snapshot: Any,
    request_id: str,
) -> tuple[AiAnalyzeOutput, str, str, str | None, str] | None:
    """Run one analyze generation and ground-validate it.

    Returns ``(output, validation_status, provider_name, provider_model, prompt_digest)``
    or ``None`` when even the deterministic fallback could not be grounded (a bug that
    must surface as a server error, never as an invalid persisted record).
    """
    provider_name = "template"
    provider_model: str | None = None
    prompt_digest = template_provider.prompt_sha256(request)
    try:
        result = provider.generate(request)
        is_amd = getattr(result, "provider", "template") == "amd"
        output = (
            _assemble_amd_analyze(snapshot, result.candidate) if is_amd else result.candidate
        )
        ok, reason = validate_analyze_output(output, snapshot)
    except Exception as exc:
        log.info(
            "AI cross-model analysis generation failed (%s, request_id=%s); using fallback",
            _provider_failure_label(exc),
            request_id,
        )
        output = fallback_output_analyze(snapshot)
        validation_status = "fallback"
    else:
        if ok:
            validation_status = "validated"
            if is_amd:
                # Provenance reflects the ACTUAL generator: an accepted AMD candidate.
                provider_name = "amd"
                provider_model = provider.model
                prompt_digest = provider.prompt_sha256(request)
        else:
            log.info("AI analysis failed grounding validation (%s); using fallback", reason)
            output = fallback_output_analyze(snapshot)
            validation_status = "fallback"

    if validation_status == "fallback":
        grounded, _ = validate_analyze_output(output, snapshot)
        if not grounded:
            output = refusal_output_analyze(snapshot)
            validation_status = "refused"
            grounded, _ = validate_analyze_output(output, snapshot)
            if not grounded:  # pragma: no cover - a bug would be a server error
                return None
    return output, validation_status, provider_name, provider_model, prompt_digest


def _assemble_amd_review(snapshot: Any, prose: Any) -> AiEvidenceReviewOutput:
    """Backend-owned structural fields + the AMD review prose.

    Every structural field, id, modality review, quality flag, uncertainty indicator,
    provenance line, reference and non-claim comes from the deterministic template review
    of the immutable snapshot; AMD only supplies ``review_text``. Provider-invented ids,
    statuses, flags or scores are therefore never trusted - and anything ungrounded in the
    prose is rejected by the unchanged grounding validator.
    """
    baseline = TemplateEvidenceReviewProvider().review(snapshot)
    return AiEvidenceReviewOutput.model_validate(
        {**baseline.model_dump(), "review_text": getattr(prose, "review_text")}
    )


def _review_with_provider(
    provider: Any,
    template_provider: Any,
    request: Any,
    snapshot: Any,
    request_id: str,
) -> tuple[AiEvidenceReviewOutput, str, str, str | None, str] | None:
    """Run one evidence-review generation and ground-validate it.

    Returns ``(output, validation_status, provider_name, provider_model, prompt_digest)``
    or ``None`` when even the deterministic fallback could not be grounded (a bug that
    must surface as a server error, never as an invalid persisted record).
    """
    provider_name = "template"
    provider_model: str | None = None
    prompt_digest = template_provider.prompt_sha256(request)
    try:
        result = provider.generate(request)
        is_amd = getattr(result, "provider", "template") == "amd"
        output = _assemble_amd_review(snapshot, result.candidate) if is_amd else result.candidate
        ok, reason = validate_review_output(output, snapshot)
    except Exception as exc:
        log.info(
            "AI evidence review generation failed (%s, request_id=%s); using fallback",
            _provider_failure_label(exc),
            request_id,
        )
        output = fallback_output_review(snapshot)
        validation_status = "fallback"
    else:
        if ok:
            validation_status = "validated"
            if is_amd:
                # Provenance reflects the ACTUAL generator: an accepted AMD candidate.
                provider_name = "amd"
                provider_model = provider.model
                prompt_digest = provider.prompt_sha256(request)
        else:
            log.info("AI review failed grounding validation (%s); using fallback", reason)
            output = fallback_output_review(snapshot)
            validation_status = "fallback"

    if validation_status == "fallback":
        grounded, _ = validate_review_output(output, snapshot)
        if not grounded:
            output = refusal_output_review(snapshot)
            validation_status = "refused"
            grounded, _ = validate_review_output(output, snapshot)
            if not grounded:  # pragma: no cover - a bug would be a server error
                return None
    return output, validation_status, provider_name, provider_model, prompt_digest


@router.post("/explain", status_code=201)
async def create_ai_explanation(request: Request, state: AppState = Depends(get_state)) -> Response:
    """Explain ONE stored assessment snapshot and persist the validated output.

    The generator is config-selectable (slice S3): the deterministic template provider
    (default) or the AMD Token Factory provider for this feature ONLY. A template
    fallback/refusal is never labelled AMD. The response is the safe subset of the
    appended AI output record; the full record (including the grounding-checked
    structured output) is retrievable via GET /api/v1/ai/outputs/{ai_output_id}.
    """
    request_id = uuid4().hex
    try:
        # 1) auth FIRST: no evidence is read or generated for an unauthorized caller.
        actor, auth_error = _authenticate(state, request, request_id)
        if auth_error is not None:
            return auth_error
        store = state.assessments
        if store is None:
            return _error(
                503,
                "storage_unavailable",
                "Assessment storage is unavailable; nothing was saved.",
                request_id,
            )
        ai_store: AiOutputStore | None = state.ai_outputs
        if ai_store is None:
            return _error(
                503, "storage_unavailable", "AI output storage is unavailable; nothing was saved.", request_id
            )

        # 2) strict request contract (exactly one assessment id; extras rejected).
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            return _error(
                422, "request_validation", "Request body is too large.", request_id,
                details={"param": "body", "reason": "too_large"},
            )
        try:
            payload: Any = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _error(
                422, "request_validation", "Request body is not valid JSON.", request_id,
                details={"param": "body", "reason": "invalid_json"},
            )
        try:
            req = AiExplainRequest.model_validate(payload)
        except ValidationError as exc:
            return _error(
                422, "request_validation", "Request failed validation.", request_id,
                details=_validation_details(exc),
            )
        try:
            assessment_id = UUID(req.assessment_id)
        except ValueError:
            return _error(
                422, "request_validation", "assessment_id must be a UUID.", request_id,
                details={"param": "assessment_id", "reason": "not_a_uuid"},
            )

        # 3) read ONLY the immutable snapshot, owner-scoped (unknown or foreign id
        #    produce the same 404: no existence leakage).
        snapshot = await run_in_threadpool(store.get, assessment_id, owner=actor)
        if snapshot is None:
            return _error(
                404, "not_found", "No such assessment.", request_id,
                details={"assessment_id": req.assessment_id},
            )

        # 4) generation + grounding validation (provider-neutral boundary). The
        #    configured provider may be AMD (explain ONLY). The persisted provider
        #    reflects the ACTUAL generator: a template fallback/refusal is never
        #    labelled AMD-generated.
        provider = get_ai_provider("explain", settings=state.settings)
        template_provider = get_ai_provider("explain", provider="template")
        request = provider.build_request((snapshot,))
        present = sum(1 for m in snapshot.modalities if m.availability == "available")
        if present == 0:
            output = refusal_output(snapshot)
            validation_status: str = "refused"
            provider_name: str = "template"
            provider_model: str | None = None
            prompt_digest: str = template_provider.prompt_sha256(request)
        else:
            outcome = _explain_with_provider(
                provider, template_provider, request, snapshot, request_id
            )
            if outcome is None:  # pragma: no cover - a bug would be a server error
                return _error(
                    500,
                    "internal_error",
                    "Unexpected server error; nothing was saved.",
                    request_id,
                )
            output, validation_status, provider_name, provider_model, prompt_digest = outcome

        # 5) append-only persistence (raw prompts/secrets/audio are never stored).
        record = AiOutputRecord(
            ai_output_id=uuid4(),
            owner=actor,
            kind="explain",
            feature="explain",
            assessment_ids=[str(assessment_id)],
            provider=provider_name,
            model=provider_model,
            prompt_version=PROMPT_VERSION,
            prompt_sha256=prompt_digest,
            output=output,
            response_text=output.explanation_text,
            validation_status=validation_status,  # type: ignore[arg-type]
            created_at=datetime.now(timezone.utc),
        )
        await run_in_threadpool(ai_store.insert, record, owner=actor)

        response = AiExplainResponse(
            feature="explain",
            ai_output_id=record.ai_output_id,
            assessment_id=output.assessment_id,
            kind=record.kind,
            explanation_text=output.explanation_text,
            validation_status=record.validation_status,
            provider=record.provider,
            model=record.model,
            prompt_version=record.prompt_version,
            referenced_assessment_ids=output.referenced_assessment_ids,
            created_at=record.created_at,
        )
        return Response(
            status_code=201,
            content=response.model_dump_json(),
            media_type="application/json",
            headers={
                "Location": f"/api/v1/ai/outputs/{record.ai_output_id}",
                "X-Request-ID": request_id,
            },
        )
    except Exception:
        log.exception("POST /ai/explain failed (request_id=%s); nothing was saved", request_id)
        return _error(500, "internal_error", "Unexpected server error; nothing was saved.", request_id)


@router.get("/outputs/{ai_output_id}")
async def get_ai_output(
    ai_output_id: str, request: Request, state: AppState = Depends(get_state)
) -> Response:
    """Fetch one stored AI output by id. Owner-scoped: others see 404, never data."""
    request_id = uuid4().hex
    try:
        actor, auth_error = _authenticate(state, request, request_id)
        if auth_error is not None:
            return auth_error
        ai_store = state.ai_outputs
        if ai_store is None:
            return _error(503, "storage_unavailable", "AI output storage is unavailable.", request_id)
        try:
            parsed = UUID(ai_output_id)
        except ValueError:
            return _error(
                422, "request_validation", "ai_output_id must be a UUID.", request_id,
                details={"param": "ai_output_id", "reason": "not_a_uuid"},
            )
        record = await run_in_threadpool(ai_store.get, parsed, owner=actor)
        if record is None:
            return _error(
                404, "not_found", "No such AI output.", request_id, details={"ai_output_id": ai_output_id}
            )
        return Response(
            status_code=200,
            content=record.model_dump_json(),
            media_type="application/json",
            headers={"X-Request-ID": request_id},
        )
    except Exception:
        log.exception("GET /ai/outputs failed (request_id=%s)", request_id)
        return _error(500, "internal_error", "Unexpected server error.", request_id)


@router.post("/evidence-review", status_code=201)
async def create_ai_evidence_review(
    request: Request, state: AppState = Depends(get_state)
) -> Response:
    """Review the evidence and uncertainty of ONE stored assessment snapshot.

    Same safety envelope as POST /ai/explain: owner resolution before any evidence is
    read, read-only use of the immutable snapshot, deterministic offline template
    provider, grounding validation before persistence, append-only storage (same
    ``ai_outputs`` table, ``kind="evidence_review"``), and 404 that cannot leak whether
    an assessment exists for another owner. No list/search endpoint in this slice.
    """
    request_id = uuid4().hex
    try:
        # 1) auth FIRST: no evidence is read or generated for an unauthorized caller.
        actor, auth_error = _authenticate(state, request, request_id)
        if auth_error is not None:
            return auth_error
        store = state.assessments
        if store is None:
            return _error(
                503,
                "storage_unavailable",
                "Assessment storage is unavailable; nothing was saved.",
                request_id,
            )
        ai_store: AiOutputStore | None = state.ai_outputs
        if ai_store is None:
            return _error(
                503,
                "storage_unavailable",
                "AI output storage is unavailable; nothing was saved.",
                request_id,
            )

        # 2) strict request contract (exactly one assessment id; extras rejected).
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            return _error(
                422, "request_validation", "Request body is too large.", request_id,
                details={"param": "body", "reason": "too_large"},
            )
        try:
            payload: Any = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _error(
                422, "request_validation", "Request body is not valid JSON.", request_id,
                details={"param": "body", "reason": "invalid_json"},
            )
        try:
            req = AiEvidenceReviewRequest.model_validate(payload)
        except ValidationError as exc:
            return _error(
                422, "request_validation", "Request failed validation.", request_id,
                details=_validation_details(exc),
            )
        try:
            assessment_id = UUID(req.assessment_id)
        except ValueError:
            return _error(
                422, "request_validation", "assessment_id must be a UUID.", request_id,
                details={"param": "assessment_id", "reason": "not_a_uuid"},
            )

        # 3) read ONLY the immutable snapshot, owner-scoped (unknown or foreign id
        #    produce the same 404: no existence leakage).
        snapshot = await run_in_threadpool(store.get, assessment_id, owner=actor)
        if snapshot is None:
            return _error(
                404, "not_found", "No such assessment.", request_id,
                details={"assessment_id": req.assessment_id},
            )

        # 4) eligibility + generation + grounding validation (provider-neutral boundary).
        #    The configured provider may be AMD (evidence_review). The persisted provider
        #    reflects the ACTUAL generator: a template fallback/refusal is never labelled AMD.
        provider = get_ai_provider("evidence_review", settings=state.settings)
        template_provider = get_ai_provider("evidence_review", provider="template")
        request = provider.build_request((snapshot,))
        provider_name: str = "template"
        provider_model: str | None = None
        prompt_digest: str = template_provider.prompt_sha256(request)
        present = sum(1 for m in snapshot.modalities if m.availability == "available")
        if present == 0:
            output: AiEvidenceReviewOutput = refusal_output_review(snapshot)
            validation_status: str = "refused"
        else:
            outcome = _review_with_provider(
                provider, template_provider, request, snapshot, request_id
            )
            if outcome is None:  # pragma: no cover - a bug would be a server error
                return _error(
                    500,
                    "internal_error",
                    "Unexpected server error; nothing was saved.",
                    request_id,
                )
            output, validation_status, provider_name, provider_model, prompt_digest = outcome

        # 5) append-only persistence (raw prompts/secrets/audio are never stored).
        record = AiOutputRecord(
            ai_output_id=uuid4(),
            owner=actor,
            kind="evidence_review",
            feature="evidence_review",
            assessment_ids=[str(assessment_id)],
            provider=provider.name,
            model=provider.model,
            prompt_version=PROMPT_VERSION_REVIEW,
            prompt_sha256=provider.prompt_sha256(request),
            output=output,
            response_text=output.review_text,
            validation_status=validation_status,  # type: ignore[arg-type]
            created_at=datetime.now(timezone.utc),
        )
        await run_in_threadpool(ai_store.insert, record, owner=actor)

        response = AiEvidenceReviewResponse(
            feature="evidence_review",
            ai_output_id=record.ai_output_id,
            assessment_id=output.assessment_id,
            kind=record.kind,
            review_text=output.review_text,
            validation_status=record.validation_status,
            provider=record.provider,
            model=record.model,
            prompt_version=record.prompt_version,
            referenced_assessment_ids=output.referenced_assessment_ids,
            created_at=record.created_at,
        )
        return Response(
            status_code=201,
            content=response.model_dump_json(),
            media_type="application/json",
            headers={
                "Location": f"/api/v1/ai/outputs/{record.ai_output_id}",
                "X-Request-ID": request_id,
            },
        )
    except Exception:
        log.exception("POST /ai/evidence-review failed (request_id=%s); nothing was saved", request_id)
        return _error(500, "internal_error", "Unexpected server error; nothing was saved.", request_id)


@router.post("/analyze", status_code=201)
async def create_ai_analysis(request: Request, state: AppState = Depends(get_state)) -> Response:
    """Cross-model analysis of ONE stored multimodal assessment snapshot.

    Same safety envelope as the other AI routes: owner resolution before any
    evidence is read, read-only use of the immutable snapshot, provider-neutral
    generation (the deterministic template provider by default, or the AMD provider
    for this feature when configured - a template fallback/refusal is never labelled
    AMD), grounding validation before persistence, and append-only storage (same
    ``ai_outputs`` table, ``kind="analyze"``, no new migration).

    Eligibility: at least two AVAILABLE modalities are required. A single-modality
    snapshot gets a deterministic refusal (persisted, never a partial analysis);
    with two or more available modalities, missing modalities are recorded as
    limiting interpretation but do not block the analysis. No score is fused and no
    causality is inferred anywhere.
    """
    request_id = uuid4().hex
    try:
        # 1) auth FIRST: no evidence is read or generated for an unauthorized caller.
        actor, auth_error = _authenticate(state, request, request_id)
        if auth_error is not None:
            return auth_error
        store = state.assessments
        if store is None:
            return _error(
                503,
                "storage_unavailable",
                "Assessment storage is unavailable; nothing was saved.",
                request_id,
            )
        ai_store: AiOutputStore | None = state.ai_outputs
        if ai_store is None:
            return _error(
                503,
                "storage_unavailable",
                "AI output storage is unavailable; nothing was saved.",
                request_id,
            )

        # 2) strict request contract (exactly one assessment id; extras rejected).
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            return _error(
                422, "request_validation", "Request body is too large.", request_id,
                details={"param": "body", "reason": "too_large"},
            )
        try:
            payload: Any = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _error(
                422, "request_validation", "Request body is not valid JSON.", request_id,
                details={"param": "body", "reason": "invalid_json"},
            )
        try:
            req = AiAnalyzeRequest.model_validate(payload)
        except ValidationError as exc:
            return _error(
                422, "request_validation", "Request failed validation.", request_id,
                details=_validation_details(exc),
            )
        try:
            assessment_id = UUID(req.assessment_id)
        except ValueError:
            return _error(
                422, "request_validation", "assessment_id must be a UUID.", request_id,
                details={"param": "assessment_id", "reason": "not_a_uuid"},
            )

        # 3) read ONLY the immutable snapshot, owner-scoped (unknown or foreign id
        #    produce the same 404: no existence leakage).
        snapshot = await run_in_threadpool(store.get, assessment_id, owner=actor)
        if snapshot is None:
            return _error(
                404, "not_found", "No such assessment.", request_id,
                details={"assessment_id": req.assessment_id},
            )

        # 4) eligibility + generation + grounding validation (provider-neutral boundary).
        #    The configured provider may be AMD (analyze). The persisted provider reflects
        #    the ACTUAL generator: a template fallback/refusal is never labelled AMD.
        provider = get_ai_provider("analyze", settings=state.settings)
        template_provider = get_ai_provider("analyze", provider="template")
        request = provider.build_request((snapshot,))
        provider_name: str = "template"
        provider_model: str | None = None
        prompt_digest: str = template_provider.prompt_sha256(request)
        present = sum(1 for m in snapshot.modalities if m.availability == "available")
        if present < 2:
            output: AiAnalyzeOutput = refusal_output_analyze(snapshot)
            validation_status: str = "refused"
        else:
            outcome = _analyze_with_provider(
                provider, template_provider, request, snapshot, request_id
            )
            if outcome is None:  # pragma: no cover - a bug would be a server error
                return _error(
                    500,
                    "internal_error",
                    "Unexpected server error; nothing was saved.",
                    request_id,
                )
            output, validation_status, provider_name, provider_model, prompt_digest = outcome

        # 5) append-only persistence (raw prompts/secrets/audio are never stored).
        record = AiOutputRecord(
            ai_output_id=uuid4(),
            owner=actor,
            kind="analyze",
            feature="analyze",
            assessment_ids=[str(assessment_id)],
            provider=provider_name,
            model=provider_model,
            prompt_version=PROMPT_VERSION_ANALYZE,
            prompt_sha256=prompt_digest,
            output=output,
            response_text=output.analysis_text,
            validation_status=validation_status,  # type: ignore[arg-type]
            created_at=datetime.now(timezone.utc),
        )
        await run_in_threadpool(ai_store.insert, record, owner=actor)

        response = AiAnalyzeResponse(
            feature="analyze",
            ai_output_id=record.ai_output_id,
            assessment_id=output.assessment_id,
            kind=record.kind,
            analysis_text=output.analysis_text,
            validation_status=record.validation_status,
            provider=record.provider,
            model=record.model,
            prompt_version=record.prompt_version,
            referenced_assessment_ids=output.referenced_assessment_ids,
            created_at=record.created_at,
        )
        return Response(
            status_code=201,
            content=response.model_dump_json(),
            media_type="application/json",
            headers={
                "Location": f"/api/v1/ai/outputs/{record.ai_output_id}",
                "X-Request-ID": request_id,
            },
        )
    except Exception:
        log.exception("POST /ai/analyze failed (request_id=%s); nothing was saved", request_id)
        return _error(500, "internal_error", "Unexpected server error; nothing was saved.", request_id)


@router.post("/report", status_code=201)
async def create_ai_maintenance_report(
    request: Request, state: AppState = Depends(get_state)
) -> Response:
    """Maintenance report over an owner's immutable assessment history for ONE machine.

    Same safety envelope as the other AI routes: owner resolution before any evidence is
    read, read-only use of the immutable snapshots, deterministic offline template
    provider, grounding validation before persistence, and append-only storage (same
    ``ai_outputs`` table, ``kind="report"``, no new migration).

    History is retrieved newest-first through the owner-scoped store, capped at 50
    (with truncation explicitly disclosed). The window is resolved against stored
    ``created_at`` values only: ``window_end`` is the newest owner-visible assessment
    ``created_at`` and ``window_start`` is ``lookback_days`` earlier - wall-clock time
    is never used as the anchor, so the same stored history yields the same report.
    No owner-visible assessment in the window produces a deterministic refusal; no score,
    p-value, confidence or fused/failure/health value is ever created.
    """
    request_id = uuid4().hex
    try:
        # 1) auth FIRST: no evidence is read or generated for an unauthorized caller.
        actor, auth_error = _authenticate(state, request, request_id)
        if auth_error is not None:
            return auth_error
        store = state.assessments
        if store is None:
            return _error(
                503,
                "storage_unavailable",
                "Assessment storage is unavailable; nothing was saved.",
                request_id,
            )
        ai_store: AiOutputStore | None = state.ai_outputs
        if ai_store is None:
            return _error(
                503, "storage_unavailable", "AI output storage is unavailable; nothing was saved.", request_id
            )

        # 2) strict request contract (machine id + bounded lookback; extras rejected).
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            return _error(
                422, "request_validation", "Request body is too large.", request_id,
                details={"param": "body", "reason": "too_large"},
            )
        try:
            payload: Any = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _error(
                422, "request_validation", "Request body is not valid JSON.", request_id,
                details={"param": "body", "reason": "invalid_json"},
            )
        try:
            req = AiReportRequest.model_validate(payload)
        except ValidationError as exc:
            return _error(
                422, "request_validation", "Request failed validation.", request_id,
                details=_validation_details(exc),
            )

        machine_id = req.machine_id
        lookback_days = req.lookback_days
        provider = get_ai_provider("report", settings=state.settings)

        # 3) read ONLY the owner-visible immutable snapshots, newest-first, capped; one
        #    extra row lets us disclose truncation inside the window truthfully.
        fetched = await run_in_threadpool(
            store.list,
            owner=actor,
            machine_id=machine_id,
            limit=REPORT_MAX_ASSESSMENTS + 1,
            offset=0,
        )

        window_start: datetime | None = None
        window_end: datetime | None = None
        history_truncated = False
        snapshots: list[Any] = []
        if fetched:
            # Window anchored on the newest stored created_at (never wall-clock time).
            window_end = fetched[0].created_at
            window_start = window_end - timedelta(days=lookback_days)
            in_window = [snapshot for snapshot in fetched if snapshot.created_at >= window_start]
            history_truncated = len(in_window) > REPORT_MAX_ASSESSMENTS
            snapshots = in_window[:REPORT_MAX_ASSESSMENTS]

        # 4) deterministic generation + grounding validation.
        if not snapshots:
            output: AiReportOutput = refusal_output_report(machine_id, lookback_days=lookback_days)
            validation_status: str = "refused"
        else:
            try:
                request = provider.build_request(
                    tuple(snapshots),
                    machine_id=machine_id,
                    window_start=window_start,
                    window_end=window_end,
                    history_truncated=history_truncated,
                    lookback_days=lookback_days,
                )
                output = provider.generate(request).candidate
            except Exception:
                log.exception("AI maintenance report failed (request_id=%s); nothing was saved", request_id)
                output = fallback_output_report(
                    snapshots,
                    machine_id=machine_id,
                    window_start=window_start,
                    window_end=window_end,
                    history_truncated=history_truncated,
                )
                validation_status = "fallback"
            else:
                ok, reason = validate_report_output(
                    output,
                    snapshots,
                    machine_id=machine_id,
                    window_start=window_start,
                    window_end=window_end,
                    history_truncated=history_truncated,
                    lookback_days=lookback_days,
                )
                if ok:
                    validation_status = "validated"
                else:
                    log.info("AI report failed grounding validation (%s); using fallback", reason)
                    output = fallback_output_report(
                        snapshots,
                        machine_id=machine_id,
                        window_start=window_start,
                        window_end=window_end,
                        history_truncated=history_truncated,
                    )
                    validation_status = "fallback"
                    ok, _ = validate_report_output(
                        output,
                        snapshots,
                        machine_id=machine_id,
                        window_start=window_start,
                        window_end=window_end,
                        history_truncated=history_truncated,
                        lookback_days=lookback_days,
                    )
                    if not ok:  # defensive: our own fallback must stay grounded
                        output = refusal_output_report(machine_id, lookback_days=lookback_days)
                        validation_status = "refused"
                        snapshots = []
                        window_start = None
                        window_end = None
                        history_truncated = False
                        ok, _ = validate_report_output(
                            output,
                            [],
                            machine_id=machine_id,
                            window_start=None,
                            window_end=None,
                            history_truncated=False,
                            lookback_days=lookback_days,
                        )
                        if not ok:  # pragma: no cover - a bug would be a server error
                            return _error(
                                500,
                                "internal_error",
                                "Unexpected server error; nothing was saved.",
                                request_id,
                            )

        # 5) append-only persistence (raw prompts/secrets/audio are never stored).
        record = AiOutputRecord(
            ai_output_id=uuid4(),
            owner=actor,
            kind="report",
            feature="report",
            assessment_ids=[str(snapshot.assessment_id) for snapshot in snapshots],
            provider=provider.name,
            model=provider.model,
            prompt_version=PROMPT_VERSION_REPORT,
            prompt_sha256=provider.prompt_sha256(
                provider.build_request(
                    tuple(snapshots),
                    machine_id=machine_id,
                    window_start=window_start,
                    window_end=window_end,
                    history_truncated=history_truncated,
                    lookback_days=lookback_days,
                )
            ),
            output=output,
            response_text=output.report_text,
            validation_status=validation_status,  # type: ignore[arg-type]
            created_at=datetime.now(timezone.utc),
        )
        await run_in_threadpool(ai_store.insert, record, owner=actor)

        response = AiReportResponse(
            feature="report",
            ai_output_id=record.ai_output_id,
            machine_id=output.machine_id,
            kind=record.kind,
            window_start=output.window_start,
            window_end=output.window_end,
            assessment_count=output.assessment_count,
            history_truncated=output.history_truncated,
            report_text=output.report_text,
            validation_status=record.validation_status,
            provider=record.provider,
            model=record.model,
            prompt_version=record.prompt_version,
            referenced_assessment_ids=output.referenced_assessment_ids,
            created_at=record.created_at,
        )
        return Response(
            status_code=201,
            content=response.model_dump_json(),
            media_type="application/json",
            headers={
                "Location": f"/api/v1/ai/outputs/{record.ai_output_id}",
                "X-Request-ID": request_id,
            },
        )
    except Exception:
        log.exception("POST /ai/report failed (request_id=%s); nothing was saved", request_id)
        return _error(500, "internal_error", "Unexpected server error; nothing was saved.", request_id)


@router.post("/ask", status_code=201)
async def create_ai_ask_answer(request: Request, state: AppState = Depends(get_state)) -> Response:
    """Answer one bounded user question from owner-visible immutable assessment evidence.

    Same safety envelope as the other AI routes: owner resolution before any evidence is
    read, read-only use of the immutable snapshots, deterministic offline template
    provider, grounding validation before persistence, and append-only storage (same
    ``ai_outputs`` table, ``kind="ask"``, no new migration).

    This slice does NOT provide free-form natural-language reasoning. The question is
    untrusted inert data: it is sanitized/normalized only to match a fixed server-owned
    supported-intent dictionary, echoed back as a bounded sanitized copy plus a digest,
    and never interpreted as an instruction. Exactly one scope is required
    (``assessment_id`` XOR ``machine_id``); ``lookback_days`` only applies to machine
    scope and is resolved against stored ``created_at`` values (never wall-clock time).
    Unsupported questions, empty context, scope/intent mismatches and missing stored
    evidence produce deterministic persisted refusals. No score, p-value, confidence,
    status, id or timestamp is created, and no fused failure probability, health score
    or risk score is produced.
    """
    request_id = uuid4().hex
    try:
        # 1) auth FIRST: no evidence is read or generated for an unauthorized caller.
        actor, auth_error = _authenticate(state, request, request_id)
        if auth_error is not None:
            return auth_error
        store = state.assessments
        if store is None:
            return _error(
                503, "storage_unavailable", "Assessment storage is unavailable; nothing was saved.", request_id
            )
        ai_store: AiOutputStore | None = state.ai_outputs
        if ai_store is None:
            return _error(
                503, "storage_unavailable", "AI output storage is unavailable; nothing was saved.", request_id
            )

        # 2) strict request contract (bounded question + one scope; extras rejected).
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            return _error(
                422, "request_validation", "Request body is too large.", request_id,
                details={"param": "body", "reason": "too_large"},
            )
        try:
            payload: Any = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _error(
                422, "request_validation", "Request body is not valid JSON.", request_id,
                details={"param": "body", "reason": "invalid_json"},
            )
        try:
            req = AiAskRequest.model_validate(payload)
        except ValidationError as exc:
            return _error(
                422, "request_validation", "Request failed validation.", request_id,
                details=_validation_details(exc),
            )

        # 3) exactly one scope (assessment_id XOR machine_id; lookback only with machine).
        has_assessment = req.assessment_id is not None
        has_machine = req.machine_id is not None
        if has_assessment and has_machine:
            return _error(
                422, "scope_mismatch",
                "Provide exactly one of assessment_id or machine_id.", request_id,
                details={"param": "scope", "reason": "ambiguous_scope"},
            )
        if not has_assessment and not has_machine:
            return _error(
                422, "scope_required",
                "Provide exactly one of assessment_id or machine_id.", request_id,
                details={"param": "scope", "reason": "scope_required"},
            )

        # 4) read ONLY the owner-visible immutable snapshots (bounded, newest-first).
        if has_assessment:
            if req.lookback_days is not None:
                return _error(
                    422, "scope_mismatch",
                    "lookback_days is allowed only with machine_id.", request_id,
                    details={"param": "lookback_days", "reason": "not_allowed_with_assessment"},
                )
            try:
                assessment_id = UUID(req.assessment_id)  # type: ignore[arg-type]
            except ValueError:
                return _error(
                    422, "request_validation", "assessment_id must be a UUID.", request_id,
                    details={"param": "assessment_id", "reason": "not_a_uuid"},
                )
            scope = "assessment"
            machine_id = None
            lookback_days: int | None = None
            window_start: datetime | None = None
            window_end: datetime | None = None
            history_truncated = False
            snapshot = await run_in_threadpool(store.get, assessment_id, owner=actor)
            if snapshot is None:
                return _error(
                    404, "not_found", "No such assessment.", request_id,
                    details={"assessment_id": req.assessment_id},
                )
            context = [snapshot]
        else:
            scope = "machine"
            machine_id = req.machine_id
            lookback_days = req.lookback_days if req.lookback_days is not None else 30
            fetched = await run_in_threadpool(
                store.list, owner=actor, machine_id=machine_id,
                limit=ASK_MAX_ASSESSMENTS + 1, offset=0,
            )
            window_start = None
            window_end = None
            history_truncated = False
            context = []
            if fetched:
                # Window anchored on the newest stored created_at (never wall-clock time).
                window_end = fetched[0].created_at
                window_start = window_end - timedelta(days=lookback_days)
                in_window = [snap for snap in fetched if snap.created_at >= window_start]
                history_truncated = len(in_window) > ASK_MAX_ASSESSMENTS
                context = in_window[:ASK_MAX_ASSESSMENTS]

        # 5) deterministic intent matching + generation + grounding validation.
        #    A deterministically refused output (unsupported question, no evidence,
        #    insufficient evidence, scope mismatch) is itself evidence-grounded but is
        #    persisted as "refused" - exactly like slices C/D. Only a supported answer
        #    that passes grounding may become "validated".
        expected_sha = question_digest(normalize_question(req.question))
        expected_echo = sanitize_echo(req.question)
        provider = get_ai_provider("ask", settings=state.settings)
        request = provider.build_request(
            context, scope=scope, question=req.question, lookback_days=lookback_days,
            window_start=window_start, window_end=window_end,
            history_truncated=history_truncated, machine_id=machine_id,
        )

        def _ground(candidate: AiAskOutput) -> tuple[bool, str]:
            return validate_ask_output(
                candidate, context, scope=scope, machine_id=machine_id,
                lookback_days=lookback_days, window_start=window_start,
                window_end=window_end, history_truncated=history_truncated,
                question_sha256=expected_sha, question_echo=expected_echo,
            )

        try:
            output: AiAskOutput = provider.generate(request).candidate
        except Exception:
            log.exception("AI ask answer failed (request_id=%s); nothing was saved", request_id)
            output = fallback_output_ask(
                context, scope=scope, question=req.question, lookback_days=lookback_days,
                window_start=window_start, window_end=window_end,
                history_truncated=history_truncated, machine_id=machine_id,
            )
            validation_status: str = "fallback"
        else:
            if not output.answered:
                # deterministic refusal: ground-check it, then persist as "refused".
                ok, reason = _ground(output)
                if not ok:
                    log.info("AI ask refusal failed grounding validation (%s); using refusal", reason)
                    output = refusal_output_ask(
                        context, scope=scope, question=req.question, lookback_days=lookback_days,
                        window_start=window_start, window_end=window_end,
                        history_truncated=history_truncated, machine_id=machine_id,
                    )
                    ok, _ = _ground(output)
                validation_status = "refused"
                if not ok:  # pragma: no cover - a bug would be a server error
                    return _error(
                        500, "internal_error", "Unexpected server error; nothing was saved.", request_id
                    )
            else:
                ok, reason = _ground(output)
                if ok:
                    validation_status = "validated"
                else:
                    log.info("AI ask answer failed grounding validation (%s); using fallback", reason)
                    output = fallback_output_ask(
                        context, scope=scope, question=req.question, lookback_days=lookback_days,
                        window_start=window_start, window_end=window_end,
                        history_truncated=history_truncated, machine_id=machine_id,
                    )
                    validation_status = "fallback"
                    ok, _ = _ground(output)
                    if not ok:  # defensive: our own fallback must stay grounded
                        output = refusal_output_ask(
                            context, scope=scope, question=req.question, lookback_days=lookback_days,
                            window_start=window_start, window_end=window_end,
                            history_truncated=history_truncated, machine_id=machine_id,
                        )
                        validation_status = "refused"
                        ok, _ = _ground(output)
                        if not ok:  # pragma: no cover - a bug would be a server error
                            return _error(
                                500, "internal_error", "Unexpected server error; nothing was saved.", request_id
                            )

        # 6) append-only persistence (raw question/secrets/audio are never stored; only a
        #    bounded sanitized echo and a digest of the normalized question are kept).
        record = AiOutputRecord(
            ai_output_id=uuid4(),
            owner=actor,
            kind="ask",
            feature="ask",
            assessment_ids=output.referenced_assessment_ids,
            provider=provider.name,
            model=provider.model,
            prompt_version=PROMPT_VERSION_ASK,
            prompt_sha256=provider.prompt_sha256(request),
            output=output,
            response_text=output.answer_text,
            validation_status=validation_status,  # type: ignore[arg-type]
            created_at=datetime.now(timezone.utc),
        )
        await run_in_threadpool(ai_store.insert, record, owner=actor)

        response = AiAskResponse(
            feature="ask",
            ai_output_id=record.ai_output_id,
            kind=record.kind,
            scope=output.scope,
            question_sha256=output.question_sha256,
            question_echo=output.question_echo,
            answered=output.answered,
            refusal_reason=output.refusal_reason,
            answer_summary=output.answer_summary,
            answer_text=output.answer_text,
            validation_status=record.validation_status,
            provider=record.provider,
            model=record.model,
            prompt_version=record.prompt_version,
            referenced_assessment_ids=output.referenced_assessment_ids,
            created_at=record.created_at,
        )
        return Response(
            status_code=201,
            content=response.model_dump_json(),
            media_type="application/json",
            headers={
                "Location": f"/api/v1/ai/outputs/{record.ai_output_id}",
                "X-Request-ID": request_id,
            },
        )
    except Exception:
        log.exception("POST /ai/ask failed (request_id=%s); nothing was saved", request_id)
        return _error(500, "internal_error", "Unexpected server error; nothing was saved.", request_id)
