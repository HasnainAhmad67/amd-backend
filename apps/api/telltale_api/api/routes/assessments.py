"""Assessment routes - slice 1: SENSOR-ONLY server-side assessments.

Server-defined profile: ``expected_modalities`` is fixed to ``["sensor"]`` by the
server (clients cannot send it). A deliberately sensor-only assessment is COMPLETE
when sensor evidence is available; it is NOT "incomplete" merely because the
product also supports acoustic/bearing models - multimodal completeness is a
separate server-defined profile for a future slice. Deliberate scope and missing
evidence are therefore never conflated.

Provenance: ``source_mode`` accepts only ``dataset_demo`` or ``simulated``. The
AI4I-style demo features are NOT claimed to be real fan/bearing measurements; the
assess/v1 rules label the stored evidence ``illustrative``. Clients submit inputs
only - inference runs on the server (threadpool) and the original model output is
stored unchanged. Failed inference never persists a successful assessment.

Demo authentication (NOT production authentication): bearer tokens come from the
TELLTALE_API_TOKENS setting (strict JSON object ``{"<actor>": "<token>", ...}``).
Empty/unconfigured, malformed, or invalid configuration => every request is
denied (fail closed). Tokens are compared with ``secrets.compare_digest`` and are
never logged or echoed. The ``X-Actor-ID`` header is ignored; caller-supplied
request ids are ignored (a fresh server id is generated per request). Cross-owner
GETs return 404 and lists are always owner-filtered. LEGACY ROUTES REMAIN
OUTSIDE this auth boundary and are unchanged.
"""

import hashlib
import io
import json
import logging
import os
import secrets
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException
from telltale_ml.acoustic.detector import MIN_SECONDS
from telltale_ml.acoustic.features import load_audio

from ...schemas.acoustic import AcousticAnalysis, FaultDiagnosis
from ...schemas.assessment import (
    AssessmentCreateRequest,
    AssessmentListResponse,
    EvidenceDraft,
    ModelIdentity,
    ModalityDraft,
    QualityFlag,
    SensorSubmission,
)
from ...schemas.sensor import SensorAssessment
from ...services.assess import create_assessment as assemble_assessment
from ...services.assessment_store import DEFAULT_PAGE, MAX_PAGE, AssessmentAlreadyExists
from ..deps import AppState, get_state

log = logging.getLogger("telltale.assessments")

router = APIRouter()

# This slice supports the FAN machine type only. Support is explicit here, never
# dynamic: another machine type is NOT enabled merely because its model happens to
# load, and there is no fallback from an unsupported type to fan.
ACOUSTIC_SLICE_MACHINE_TYPE = "fan"
# Filename allowlist (display-trust only; decoding must still succeed - an extension
# or MIME type alone is never trusted). MATLAB (.mat) is rejected in this slice.
AUDIO_UPLOAD_EXTENSIONS = {".wav", ".flac", ".ogg"}
MAT_MAGIC = b"MATLAB"
# Strict multipart contract: exactly these parts, at most one file.
_FORM_FIELDS = {"file", "machine_id", "machine_type", "observed_at"}
# Strict multipart contract for POST /assessments/multimodal: exactly these parts;
# each modality has its OWN part name, so one part can never serve two modalities
# (unknown, duplicate and malformed parts are rejected, never ignored).
_MULTIMODAL_FORM_FIELDS = {"machine_id", "sensor_json", "acoustic_file", "bearing_file", "observed_at"}
# At most one file per audio modality: acoustic_file + bearing_file.
_MULTIMODAL_MAX_FILES = 2
# Text parts: machine_id + sensor_json + observed_at. The parser limit keeps one
# extra slot so duplicate/unknown text parts reach the endpoint's PRECISE
# duplicate/unknown error (instead of the parser's generic limit message).
_MULTIMODAL_MAX_FIELDS = 4
# Non-file form parts are bounded by the multipart parser to this size (bytes).
# NOTE: the parser's per-part cap covers TEXT fields only; the file part is spooled
# during parsing and is bounded per-file by the checks in _bounded_upload_bytes.
# Total request-body size must be bounded at the deployment level (reverse proxy /
# host) because uvicorn applies no body-size limit - see the route docstring.
_MULTIPART_FIELD_LIMIT = 16 * 1024
# Overhead slack (boundary + headers) added to max_upload_mb for parser-level checks
# and for the Content-Length fast-reject heuristic.
_MULTIPART_OVERHEAD = 16 * 1024


# ---------------------------------------------------------------- error envelope (new routes only)


def _error(status: int, code: str, message: str, request_id: str, details: Any = None) -> JSONResponse:
    body: dict[str, Any] = {"error": {"code": code, "message": message}, "request_id": request_id}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body, headers={"X-Request-ID": request_id})


class _BodyProblem(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _ParamProblem(Exception):
    def __init__(self, param: str, reason: str) -> None:
        super().__init__(param)
        self.param = param
        self.reason = reason


class _UploadProblem(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _sanitize_filename(filename: str | None) -> str:
    """Keep only a client filename's final path component for display."""
    return Path((filename or "").replace("\\", "/")).name


async def _read_json(request: Request) -> Any:
    if "application/json" not in request.headers.get("content-type", ""):
        raise _BodyProblem("invalid_content_type")
    try:
        return json.loads(await request.body())
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise _BodyProblem("invalid_json")


def _validation_details(exc: ValidationError) -> list[dict[str, Any]]:
    # Sanitized: loc/msg/type only - no echoed input values, no URLs, no context.
    details: list[dict[str, Any]] = []
    for err in exc.errors():
        details.append(
            {
                "loc": [str(part) for part in err.get("loc", ())],
                "msg": str(err.get("msg", "")),
                "type": str(err.get("type", "")),
            }
        )
    return details


def _parse_page_int(raw: str | None, param: str, default: int, lo: int, hi: int | None) -> int:
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise _ParamProblem(param, "not_an_integer")
    if value < lo or (hi is not None and value > hi):
        raise _ParamProblem(param, "out_of_range")
    return value


# ---------------------------------------------------------------- demo authentication


class _DuplicateKey(Exception):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _DuplicateKey(key)
        out[key] = value
    return out


@lru_cache
def _parse_api_tokens(raw: str) -> tuple[dict[str, str], str | None]:
    """Parse TELLTALE_API_TOKENS (strict JSON object: actor -> bearer token).

    Returns ``(mapping, error_code)``. Error codes never contain token values and
    the raw configuration is never logged. Invalid configuration fails CLOSED.
    """
    if not raw.strip():
        return {}, None  # unconfigured -> deny (auth_disabled)
    try:
        data = json.loads(raw, object_pairs_hook=_strict_object)
    except _DuplicateKey:
        return {}, "duplicate_actor"
    except json.JSONDecodeError:
        return {}, "malformed_json"
    if not isinstance(data, dict):
        return {}, "not_an_object"
    actors: dict[str, str] = {}
    seen_tokens: set[str] = set()
    for actor, token in data.items():
        if not isinstance(actor, str) or not actor.strip():
            return {}, "invalid_actor"
        if not isinstance(token, str) or not token.strip():
            return {}, "invalid_token"
        if token in seen_tokens:
            return {}, "duplicate_token"
        seen_tokens.add(token)
        actors[actor] = token
    return actors, None


_logged_config_errors: set[str] = set()


def _log_config_error(code: str) -> None:
    if code not in _logged_config_errors:
        _logged_config_errors.add(code)
        # Only the error kind is logged - never the raw setting, never tokens.
        log.error(
            "TELLTALE_API_TOKENS is invalid (%s); assessment routes deny all requests",
            code,
        )


def _authenticate(
    state: AppState, request: Request, request_id: str
) -> tuple[str | None, JSONResponse | None]:
    """Returns (actor, None) on success or (None, error_response). Default-deny."""
    # Demo mode: if TELLTALE_DEMO_MODE=1 env var is set, enable fixed demo owner.
    # When disabled, the normal auth is enforced strictly.
    demo_mode = os.environ.get("TELLTALE_DEMO_MODE", "").strip() in ("1", "true", "True")
    if demo_mode:
        mapping = {"local-demo-user": "demo-token-123"}
    else:
        mapping, config_error = _parse_api_tokens(state.settings.api_tokens)
        if config_error is not None:
            _log_config_error(config_error)
            return None, _error(
                401,
                "auth_misconfigured",
                "Assessment authentication is misconfigured; requests are denied. "
                "Ask the operator to fix the TELLTALE_API_TOKENS setting.",
                request_id,
            )
        if not mapping:
            return None, _error(
                401,
                "auth_disabled",
                "The assessment API is disabled: no API tokens are configured (set TELLTALE_API_TOKENS).",
                request_id,
            )
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None, _error(401, "unauthorized", "Missing bearer token.", request_id)
    candidate = header[len("Bearer "):].strip().encode("utf-8")
    for actor, token in mapping.items():
        if secrets.compare_digest(candidate, token.encode("utf-8")):
            return actor, None
    return None, _error(401, "unauthorized", "Invalid bearer token.", request_id)


async def _bounded_upload_bytes(upload: UploadFile, limit: int) -> bytes:
    """Read at most ``limit + 1`` bytes - never trust Content-Length or MIME type.

    Raises :class:`_UploadProblem` on oversize/empty input. The upload resource is
    always closed, on success and on every error path.
    """
    try:
        data = await upload.read(limit + 1)
        if len(data) > limit:
            raise _UploadProblem(
                413,
                "file_too_large",
                f"The upload exceeds the {limit // (1024 * 1024)} MB limit.",
            )
        if not data:
            raise _UploadProblem(422, "invalid_audio", "The uploaded audio file is empty.")
        return data
    finally:
        await upload.close()


def _analyse_buffer(detector, buf: io.BytesIO) -> tuple[dict[str, Any], float | None]:
    """Decode-validate-score ONE buffer under the detector's own effective rules.

    Bounded work only: the original duration comes from a header-only probe
    (never a full decode just to measure), the finite-sample validation decodes
    exactly the audio the model will score (truncated to the processing cap),
    and scoring itself is the UNCHANGED detector call. Raises
    :class:`_UploadProblem` for undecodable, empty, too-short or non-finite audio;
    decoder internals stay in the server-side log, never in the client message.
    """
    try:
        info = sf.info(buf)  # header-only: bounded, no full decode
    except Exception as exc:
        log.info("Acoustic upload header probe failed: %s", type(exc).__name__)
        raise _UploadProblem(
            422, "invalid_audio", "The upload could not be decoded as WAV/FLAC/OGG audio."
        ) from exc
    original_s: float | None = None
    if info.frames > 0 and info.samplerate > 0:
        original_s = float(info.frames) / float(info.samplerate)  # reliably obtainable
    try:
        buf.seek(0)
        y = load_audio(buf, detector.cfg.sample_rate, detector.cfg.channel, detector.metadata.get("max_audio_seconds"))
    except Exception as exc:
        log.info("Acoustic upload decode failed: %s", type(exc).__name__)
        raise _UploadProblem(
            422, "invalid_audio", "The upload could not be decoded as WAV/FLAC/OGG audio."
        ) from exc
    if not bool(np.isfinite(y).all()):
        raise _UploadProblem(422, "invalid_audio", "The audio contains non-finite samples and was rejected.")
    if len(y) / detector.cfg.sample_rate < MIN_SECONDS:
        raise _UploadProblem(
            422, "invalid_audio", f"The audio is too short; at least {MIN_SECONDS:.1f} s is required."
        )
    try:
        buf.seek(0)
        result = detector.analyze(buf)
    except ValueError as exc:  # detector input validation (audio-caused; sanitized)
        log.info("Acoustic analysis rejected upload: %s", type(exc).__name__)
        raise _UploadProblem(
            422, "invalid_audio", "The upload could not be analysed as WAV/FLAC/OGG audio."
        ) from exc
    # RuntimeError (e.g. ONNX session failure) is an INFERENCE failure, not bad
    # audio: it propagates to the generic handler -> sanitized 500, nothing saved.
    return result, original_s


def _diagnose_buffer(classifier, buf: io.BytesIO, rpm: float | None) -> tuple[dict[str, Any], float | None]:
    """Decode-validate-score ONE buffer under the classifier's own effective rules.

    Reuses only rules that exist on the original bearing upload path (nothing is
    invented): the ORIGINAL duration comes from a header-only probe when the
    container has a readable header - it is BEST EFFORT here because MATLAB .mat
    (the bearing dataset's own format) has no libsndfile header, so a failed probe
    yields no original duration but never rejects the upload; the decode/validation
    uses the SAME load_audio call the classifier performs (anything libsndfile
    decodes plus MATLAB .mat, model-side resample/channel-select, model processing
    cap); the minimum length is the classifier's own configured ``segment_seconds``
    (segment_features raises the same condition during scoring); finite-sample
    validation decodes exactly the audio the model will score; scoring itself is
    the UNCHANGED classifier call. Raises :class:`_UploadProblem` for undecodable,
    empty, too-short or non-finite audio; decoder internals stay in the server-side
    log, never in the client message.
    """
    original_s: float | None = None
    try:
        info = sf.info(buf)  # header-only: bounded, no full decode
        if info.frames > 0 and info.samplerate > 0:
            original_s = float(info.frames) / float(info.samplerate)  # reliably obtainable
    except Exception as exc:  # .mat and headerless containers: duration unknown, not fatal
        log.info("Bearing upload header probe unavailable: %s", type(exc).__name__)
    try:
        buf.seek(0)
        y = load_audio(
            buf,
            classifier.cfg.sample_rate,
            classifier.cfg.channel,
            classifier.metadata.get("max_audio_seconds"),
        )
    except Exception as exc:
        log.info("Bearing upload decode failed: %s", type(exc).__name__)
        raise _UploadProblem(
            422, "invalid_audio", "The upload could not be decoded as audio (WAV/FLAC/OGG or MATLAB .mat)."
        ) from exc
    if y.size == 0:
        raise _UploadProblem(422, "invalid_audio", "The uploaded audio file is empty.")
    if not bool(np.isfinite(y).all()):
        raise _UploadProblem(422, "invalid_audio", "The audio contains non-finite samples and was rejected.")
    min_seconds = float(getattr(classifier.cfg, "segment_seconds", 0.0) or 0.0)
    if min_seconds and len(y) / classifier.cfg.sample_rate < min_seconds:
        # Same condition segment_features enforces during scoring; the value comes
        # from the DEPLOYED model's configuration, never from a constant here.
        raise _UploadProblem(
            422, "invalid_audio", f"The audio is too short; at least {min_seconds:.1f} s is required."
        )
    try:
        buf.seek(0)
        result = classifier.diagnose(buf, rpm)
    except ValueError as exc:  # classifier input validation (audio-caused; sanitized)
        log.info("Bearing diagnosis rejected upload: %s", type(exc).__name__)
        raise _UploadProblem(
            422, "invalid_audio", "The upload could not be analysed as audio (WAV/FLAC/OGG or MATLAB .mat)."
        ) from exc
    # RuntimeError (e.g. LightGBM failure) is an INFERENCE failure, not bad audio:
    # it propagates to the generic handler -> sanitized 500, nothing saved.
    return result, original_s


# ---------------------------------------------------------------- routes


@router.post("", status_code=201)
async def create_assessment(request: Request, state: AppState = Depends(get_state)) -> Response:
    """Create a sensor-only assessment (trusted server-side inference).

    The server fixes expected modalities to ``["sensor"]``; completeness refers to
    that deliberate scope only. ``source_mode`` is dataset_demo/simulated demo
    provenance - AI4I-style features are illustrative, never real fan/bearing
    measurements. Flow: auth -> validation -> inference (threadpool) -> snapshot
    -> committed insert -> 201 with a Location header.
    """
    request_id = uuid4().hex
    try:
        actor, auth_error = _authenticate(state, request, request_id)
        if auth_error is not None:
            return auth_error
        body = AssessmentCreateRequest.model_validate(await _read_json(request))
        if body.machine_id not in state.fleet.machines:
            return _error(
                422,
                "unknown_machine",
                "machine_id is not a machine in the configured demo fleet.",
                request_id,
                details={"machine_id": body.machine_id},
            )
        store = state.assessments
        if store is None:
            return _error(
                503,
                "storage_unavailable",
                "Assessment storage is unavailable; nothing was saved.",
                request_id,
            )
        predictor = state.registry.require_sensor()  # raises HTTPException(503, hint) when missing
        [result] = await run_in_threadpool(predictor.assess, [body.reading.model_dump()])
        identity = state.registry.sensor_identity
        snapshot = assemble_assessment(
            machine_id=body.machine_id,
            source_mode=body.source_mode,
            expected_modalities=["sensor"],  # server-defined sensor-only profile
            modalities=[
                ModalityDraft(
                    kind="sensor",
                    availability="available",
                    reason_code="ok",
                    source_mode=body.source_mode,
                    observed_at=body.observed_at,
                    inference_at=datetime.now(timezone.utc),
                    model_identity=ModelIdentity.model_validate(identity) if identity else None,
                    limitations=list(predictor.metadata.get("caveats", [])),
                    output=SensorAssessment.model_validate(result),
                )
            ],
            evidence=[
                EvidenceDraft(
                    kind="failure_probability",
                    value=result["failure_probability"],
                    unit="probability",
                    source="sensor",
                    source_mode=body.source_mode,
                    observed_at=body.observed_at,
                )
            ],
        )
        await run_in_threadpool(store.insert, snapshot, owner=actor)
        return Response(
            status_code=201,
            content=snapshot.model_dump_json(),
            media_type="application/json",
            headers={
                "Location": f"/api/v1/assessments/{snapshot.assessment_id}",
                "X-Request-ID": request_id,
            },
        )
    except _BodyProblem as exc:
        message = (
            "Request body must be a JSON object."
            if exc.code == "invalid_content_type"
            else "Request body is not valid JSON."
        )
        return _error(422, exc.code, message, request_id)
    except ValidationError as exc:
        return _error(
            422,
            "request_validation",
            "Request failed validation.",
            request_id,
            details=_validation_details(exc),
        )
    except HTTPException as exc:  # require_sensor -> 503 with the static training hint
        code = "dependency_unavailable" if exc.status_code == 503 else "dependency_error"
        return _error(exc.status_code, code, str(exc.detail), request_id)
    except AssessmentAlreadyExists:
        return _error(409, "assessment_conflict", "The assessment already exists.", request_id)
    except Exception:
        log.exception("POST /assessments failed (request_id=%s); nothing was saved", request_id)
        return _error(500, "internal_error", "Unexpected server error; nothing was saved.", request_id)


@router.get("")
async def list_assessments(
    request: Request,
    limit: str | None = None,
    offset: str | None = None,
    machine_id: str | None = None,
    state: AppState = Depends(get_state),
) -> Response:
    """Owner-scoped, bounded listing (stable order: newest first). Read-only."""
    request_id = uuid4().hex
    try:
        actor, auth_error = _authenticate(state, request, request_id)
        if auth_error is not None:
            return auth_error
        store = state.assessments
        if store is None:
            return _error(503, "storage_unavailable", "Assessment storage is unavailable.", request_id)
        limit_i = _parse_page_int(limit, "limit", default=DEFAULT_PAGE, lo=1, hi=MAX_PAGE)
        offset_i = _parse_page_int(offset, "offset", default=0, lo=0, hi=None)
        items = await run_in_threadpool(
            store.list, owner=actor, machine_id=machine_id, limit=limit_i, offset=offset_i
        )
        envelope = AssessmentListResponse(items=items, limit=limit_i, offset=offset_i)
        return Response(
            content=envelope.model_dump_json(),
            media_type="application/json",
            headers={"X-Request-ID": request_id},
        )
    except _ParamProblem as exc:
        return _error(
            422,
            "request_validation",
            "Invalid query parameter.",
            request_id,
            details={"param": exc.param, "reason": exc.reason},
        )
    except Exception:
        log.exception("GET /assessments failed (request_id=%s)", request_id)
        return _error(500, "internal_error", "Unexpected server error.", request_id)


@router.get("/{assessment_id}")
async def get_assessment(
    assessment_id: str, request: Request, state: AppState = Depends(get_state)
) -> Response:
    """Fetch one assessment by id. Owner-scoped: other owners see 404, never data."""
    request_id = uuid4().hex
    try:
        actor, auth_error = _authenticate(state, request, request_id)
        if auth_error is not None:
            return auth_error
        store = state.assessments
        if store is None:
            return _error(503, "storage_unavailable", "Assessment storage is unavailable.", request_id)
        snapshot = await run_in_threadpool(store.get, assessment_id, owner=actor)
        if snapshot is None:
            return _error(
                404,
                "not_found",
                "No such assessment.",
                request_id,
                details={"assessment_id": assessment_id},
            )
        return Response(
            content=snapshot.model_dump_json(),
            media_type="application/json",
            headers={"X-Request-ID": request_id},
        )
    except Exception:
        log.exception("GET /assessments/{id} failed (request_id=%s)", request_id)
        return _error(500, "internal_error", "Unexpected server error.", request_id)


@router.post("/acoustic", status_code=201)
async def create_acoustic_assessment(request: Request, state: AppState = Depends(get_state)) -> Response:
    """Create an ACOUSTIC-ONLY assessment from an uploaded recording (trusted server-side inference).

    Server-defined profile: ``expected_modalities`` is fixed to ``["acoustic"]``;
    completeness means "acoustic evidence is present for this acoustic-only
    scope" - NOT multimodal completeness. This slice supports the FAN machine
    type only: the selected machine's acoustic profile must be ``fan`` (no
    fallback to fan from other types; no other type is enabled merely because
    its model happens to load).

    Provenance: ``source_mode`` is server-fixed to ``uploaded`` - an upload is an
    upload, independent of the dataset the MODEL was trained on
    (``model_identity.training_dataset``, e.g. mimii_due). An uploaded recording
    is NOT automatically illustrative, and a filename is NEVER trusted ground
    truth. Retention is HASH-ONLY: the raw bytes are fingerprinted
    (``recording_sha256``) and then discarded - the hash is an identity/integrity
    record, NOT replay capability. The model's 30 s processing cap is disclosed
    through original/processed duration evidence and an ``AUDIO_TRUNCATED``
    quality flag when truncation occurred; scoring behaviour is unchanged.
    Conformal p-values are only comparable to the calibration recordings; no
    unconditional validity is claimed under recording-condition shift.

    Upload safety: auth is validated BEFORE the multipart body is parsed; the
    parser is bounded (max_files=1, max_fields=4, text parts capped); the
    endpoint then re-checks size with a bounded read (Content-Length is never
    trusted - it may be absent or false); the extension allowlist is backed by
    successful libsndfile decoding; MATLAB (.mat) recordings are rejected here.
    DEPLOYMENT-LEVEL REQUEST LIMITS (not enforceable in-app): uvicorn applies no
    total request-body cap, and the multipart parser spools the file part to the
    process TEMP directory while parsing - deployments accepting untrusted traffic
    must bound request size at the reverse proxy/host and keep TEMP on a
    writable, quota'd volume (D: in this deployment).

    Flow: auth -> bounded strict multipart validation -> threadpool inference ->
    snapshot -> committed insert -> 201 + Location. Failed validation, inference
    or insertion never persists an assessment; upload resources are closed on
    every path.
    """
    request_id = uuid4().hex
    upload: UploadFile | None = None
    try:
        # 1) auth FIRST: the multipart body is never parsed for an unauthorized caller.
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
        limit = int(state.settings.max_upload_mb * 1024 * 1024)
        field_limit = min(_MULTIPART_FIELD_LIMIT, limit + _MULTIPART_OVERHEAD)
        # 2) Fast rejection ONLY when the declared length is absurdly large. A
        # small/absent/false Content-Length is never trusted: the bounded read in
        # _bounded_upload_bytes is the authoritative per-file check.
        raw_length = request.headers.get("content-length", "")
        if raw_length.isdigit() and int(raw_length) > limit + _MULTIPART_OVERHEAD:
            return _error(
                413,
                "file_too_large",
                f"The upload exceeds the {int(state.settings.max_upload_mb)} MB limit.",
                request_id,
            )
        # 3) Bounded strict multipart parse (text parts capped; at most one file,
        # at most four fields - anything else is rejected, never ignored).
        try:
            form = await request.form(max_files=1, max_fields=4, max_part_size=field_limit)
        except MultiPartException as exc:  # raised directly by some parser paths
            log.info("Acoustic upload rejected by parser limits: %s", str(exc)[:120])
            return _error(422, "request_validation", "Malformed or oversized multipart form data.", request_id)
        except StarletteHTTPException as exc:  # starlette wraps parser-limit errors as HTTP 400
            log.info("Acoustic upload rejected by parser limits: %s", str(exc.detail)[:120])
            return _error(422, "request_validation", "Malformed or oversized multipart form data.", request_id)
        except UnicodeDecodeError:
            return _error(422, "request_validation", "Multipart form data is not decodable.", request_id)
        items = form.multi_items()
        counts: dict[str, int] = {}
        for key, _value in items:
            counts[key] = counts.get(key, 0) + 1
        duplicates = sorted(key for key, n in counts.items() if n > 1)
        unknown = sorted(key for key in counts if key not in _FORM_FIELDS)
        if duplicates or unknown:
            return _error(
                422,
                "request_validation",
                "Form fields must be unique and within the allowed set "
                "(file, machine_id, machine_type, observed_at).",
                request_id,
                details={"duplicate": duplicates, "unknown": unknown},
            )
        parts = {key: value for key, value in items}
        candidate = parts.get("file")
        if not isinstance(candidate, UploadFile):
            return _error(
                422,
                "request_validation",
                "Exactly one uploaded audio file is required (form field 'file').",
                request_id,
                details={"param": "file"},
            )
        upload = candidate
        machine_id = parts.get("machine_id")
        machine_type = parts.get("machine_type")
        observed_raw = parts.get("observed_at")
        if not isinstance(machine_id, str) or not machine_id:
            return _error(422, "request_validation", "machine_id is required.", request_id, details={"param": "machine_id"})
        if machine_type is not None and (not isinstance(machine_type, str) or not machine_type):
            return _error(
                422, "request_validation", "machine_type must be a non-empty string.", request_id,
                details={"param": "machine_type"},
            )
        if observed_raw is not None and not isinstance(observed_raw, str):
            return _error(
                422, "request_validation", "observed_at must be an ISO 8601 string.", request_id,
                details={"param": "observed_at"},
            )
        # 4) Observed timestamp: timezone-aware ISO only (naive rejected, never
        # fabricated); omitted stays None and is surfaced by TIMESTAMP_MISSING.
        observed_at = None
        if observed_raw:
            try:
                parsed_dt = datetime.fromisoformat(observed_raw.replace("Z", "+00:00"))
            except ValueError:
                parsed_dt = None
            if parsed_dt is None or parsed_dt.tzinfo is None:
                return _error(
                    422,
                    "request_validation",
                    "observed_at must be a timezone-aware ISO 8601 timestamp (naive timestamps are rejected).",
                    request_id,
                    details={"param": "observed_at"},
                )
            observed_at = parsed_dt.astimezone(timezone.utc)
        # 5) Machine scope: the profile must be fan (explicit slice boundary).
        machine = state.fleet.machines.get(machine_id)
        if machine is None:
            return _error(
                422,
                "unknown_machine",
                "machine_id is not a machine in the configured demo fleet.",
                request_id,
                details={"machine_id": machine_id},
            )
        profile = machine.spec.acoustic_profile
        if not profile:
            return _error(
                422,
                "acoustic_profile_mismatch",
                "The selected machine has no acoustic profile; acoustic assessments need one.",
                request_id,
                details={"machine_id": machine_id},
            )
        if machine_type is not None and machine_type != profile:
            return _error(
                422,
                "acoustic_profile_mismatch",
                "machine_type does not match the selected machine's acoustic profile.",
                request_id,
                details={"machine_type": machine_type, "acoustic_profile": profile},
            )
        if profile != ACOUSTIC_SLICE_MACHINE_TYPE:
            return _error(
                422,
                "unsupported_machine_type",
                "This assessment endpoint supports the 'fan' machine type only in this slice.",
                request_id,
                details={"acoustic_profile": profile},
            )
        detector = state.registry.require_acoustic(ACOUSTIC_SLICE_MACHINE_TYPE)
        identity = state.registry.acoustic_identity.get(ACOUSTIC_SLICE_MACHINE_TYPE)
        reference_fp = (identity or {}).get("reference_fingerprint")
        # 6) Mutable enrolled references can never be identified/snapshotted safely
        # here: refuse instead of recording a bare condition name as provenance.
        if getattr(detector, "enrolled", None):
            return _error(
                409,
                "enrolled_reference_active",
                "This sound model has enrolled conditions; the exact enrolled reference state cannot be "
                "identified and snapshotted safely in this slice, so no assessment was created.",
                request_id,
                details={"enrolled": sorted(getattr(detector, "enrolled", {}))},
            )
        # 7) Bounded read (authoritative size check) + resource cleanup on every path.
        filename = _sanitize_filename(upload.filename)
        data = await _bounded_upload_bytes(upload, limit)
        upload = None  # closed by _bounded_upload_bytes
        if not filename or Path(filename).suffix.lower() not in AUDIO_UPLOAD_EXTENSIONS:
            return _error(
                422, "invalid_audio", "Unsupported audio format; upload a WAV, FLAC or OGG file.", request_id
            )
        if data[: len(MAT_MAGIC)] == MAT_MAGIC:
            return _error(
                422, "invalid_audio", "MATLAB recordings are not accepted here; upload WAV, FLAC or OGG.", request_id
            )
        # 8) Threadpool: header probe + finite-sample validation + UNCHANGED scoring.
        result, original_s = await run_in_threadpool(_analyse_buffer, detector, io.BytesIO(data))
        processed_s = float(result["duration_s"])
        cap = detector.metadata.get("max_audio_seconds")
        truncated = original_s is not None and cap is not None and processed_s < original_s - 0.05
        extra_flags: list[QualityFlag] = []
        if truncated:
            extra_flags.append(
                QualityFlag(
                    code="AUDIO_TRUNCATED",
                    severity="info",
                    explanation=f"The recording is about {original_s:.1f} s but the sound model processes at most "
                    f"{cap} s; only the first {processed_s:.1f} s were scored (scoring behaviour unchanged).",
                )
            )
        # 9) Snapshot: server-owned everything; typed ML output preserved as-is
        # (filename added as sanitized untrusted display metadata, like the legacy route).
        output_doc = dict(result)
        output_doc["filename"] = filename
        evidence = [
            EvidenceDraft(
                kind="anomaly_score",
                value=result["score"],
                unit="score",  # higher = more unusual than the normal reference
                source="acoustic",
                source_mode="uploaded",
                observed_at=observed_at,
            )
        ]
        if result.get("p_value") is not None:
            evidence.append(
                EvidenceDraft(
                    kind="conformal_p_value",
                    value=result["p_value"],
                    unit=None,  # share of normal calibration recordings at least this unusual
                    source="acoustic",
                    source_mode="uploaded",
                    observed_at=observed_at,
                )
            )
        if original_s is not None:
            evidence.append(
                EvidenceDraft(
                    kind="audio_duration_original_s",
                    value=original_s,
                    unit="seconds",  # header-derived original length, before the processing cap
                    source="acoustic",
                    source_mode="uploaded",
                    observed_at=observed_at,
                )
            )
        evidence.append(
            EvidenceDraft(
                kind="audio_duration_processed_s",
                value=processed_s,
                unit="seconds",  # what the model actually scored
                source="acoustic",
                source_mode="uploaded",
                observed_at=observed_at,
            )
        )
        identity_public = (
            {key: value for key, value in identity.items() if key in ModelIdentity.model_fields}
            if identity
            else None
        )
        snapshot = assemble_assessment(
            machine_id=machine_id,
            source_mode="uploaded",  # server-fixed: an upload is an upload
            expected_modalities=["acoustic"],  # server-defined acoustic-only profile
            modalities=[
                ModalityDraft(
                    kind="acoustic",
                    availability="available",
                    reason_code="ok",
                    source_mode="uploaded",
                    observed_at=observed_at,
                    inference_at=datetime.now(timezone.utc),
                    model_identity=ModelIdentity.model_validate(identity_public) if identity_public else None,
                    recording_id=str(uuid4()),
                    recording_sha256=hashlib.sha256(data).hexdigest(),
                    reference_id=f"sha256:{reference_fp}" if reference_fp else None,
                    limitations=list(detector.metadata.get("caveats", [])),
                    output=AcousticAnalysis.model_validate(output_doc),
                )
            ],
            evidence=evidence,
            extra_flags=extra_flags,
        )
        await run_in_threadpool(store.insert, snapshot, owner=actor)
        return Response(
            status_code=201,
            content=snapshot.model_dump_json(),
            media_type="application/json",
            headers={
                "Location": f"/api/v1/assessments/{snapshot.assessment_id}",
                "X-Request-ID": request_id,
            },
        )
    except _UploadProblem as exc:
        return _error(exc.status, exc.code, exc.message, request_id)
    except HTTPException as exc:  # require_acoustic -> 503/404 with the static training hint
        code = "dependency_unavailable" if exc.status_code == 503 else "dependency_error"
        return _error(exc.status_code, code, str(exc.detail), request_id)
    except AssessmentAlreadyExists:
        return _error(409, "assessment_conflict", "The assessment already exists.", request_id)
    except Exception:
        log.exception("POST /assessments/acoustic failed (request_id=%s); nothing was saved", request_id)
        return _error(500, "internal_error", "Unexpected server error; nothing was saved.", request_id)
    finally:
        if upload is not None:  # belt-and-braces: closed on every remaining path too
            try:
                await upload.close()
            except Exception:
                pass


@router.post("/bearing", status_code=201)
async def create_bearing_assessment(request: Request, state: AppState = Depends(get_state)) -> Response:
    """Create a BEARING-ONLY assessment from an uploaded recording (trusted server-side inference).

    Server-defined profile: ``expected_modalities`` is fixed to ``["bearing"]``;
    completeness means "bearing evidence is present for this bearing-only scope".
    This slice runs exactly ONE model - the deployed bearing fault classifier -
    for any fleet machine: no fallback to a fan/acoustic model ever happens, no
    other modality or machine type is enabled dynamically, and no sensor+acoustic
    probability is fused (the overall status comes from the conservative
    assess/v1 bearing rules alone, which never report ``critical``).

    ``machine_type`` is optional request metadata verified for consistency with
    the machine's ``acoustic_profile`` recorded in the demo fleet (a contradiction
    is rejected as forged/inconsistent); it NEVER selects a model here.

    Provenance: ``source_mode`` is server-fixed to ``uploaded`` - an upload is an
    upload, independent of the dataset the MODEL was trained on
    (``model_identity.training_dataset``; the Mendeley bearing set). Retention is
    HASH-ONLY: raw bytes are fingerprinted (``recording_sha256``) and then
    discarded - the hash is an identity record, NOT replay capability. The
    sanitized filename is untrusted DISPLAY metadata only: it is never a storage
    path, an instruction or a ground-truth label (labels come from the model's
    typed output). Observation time is separate from inference time; a missing
    ``observed_at`` stays missing (TIMESTAMP_MISSING), never fabricated.

    Bearing input validation (existing rules only - none invented): size is
    bounded by ``max_upload_mb`` (bounded read; Content-Length is only a fast
    reject when absurdly large, never trusted when small/absent/lying); empty
    input is rejected; decoding must succeed through the classifier's OWN
    load_audio path - anything libsndfile decodes (WAV/FLAC/OGG, ...) and MATLAB
    .mat (the bearing dataset's native format, deliberately ACCEPTED here, unlike
    the fan slice) are supported, everything else is rejected; non-finite samples
    are rejected; audio shorter than the DEPLOYED model's configured
    ``segment_seconds`` is rejected (same condition its own feature extraction
    raises). The original upload path defines NO client-side sample-rate,
    channel-count, duration-ceiling or extension-allowlist rule: none is added
    here (decode success is the criterion) and that gap is reported rather than
    silently invented. The model's ``max_audio_seconds`` processing cap is
    disclosed through original/processed duration evidence plus an
    ``AUDIO_TRUNCATED`` quality flag when truncation actually occurred (original
    duration is header-derived and simply unavailable for .mat - nothing is
    fabricated); scoring behaviour is unchanged. Shaft speed is never a client
    field in this contract: the machine's live telemetry rpm is reused exactly
    like the legacy /acoustic/diagnose machine_id path, and when it is unknown
    the model's configured default is assumed and reported (``rpm_assumed``).
    Conformal prediction sets/coverage are only comparable to the calibration
    recordings; no unconditional validity is claimed under condition shift.

    Upload safety: auth is validated BEFORE the multipart body is parsed; the
    parser is bounded (max_files=1, max_fields=4, text parts capped); the
    endpoint then re-checks size with a bounded read; unknown and duplicate form
    fields are rejected (clients cannot smuggle scores, ids, owner, references,
    model identities, source_mode or expected modalities). DEPLOYMENT-LEVEL
    REQUEST LIMITS (not enforceable in-app): uvicorn applies no total
    request-body cap and the multipart parser spools the file part to the
    process TEMP directory while parsing - deployments accepting untrusted
    traffic must bound request size at the reverse proxy/host and keep TEMP on a
    writable, quota'd volume (D: in this deployment).

    Flow: auth -> bounded strict multipart validation -> threadpool inference ->
    snapshot -> committed insert -> 201 + Location. Failed validation, inference
    or insertion never persists an assessment; upload resources are closed on
    every path.
    """
    request_id = uuid4().hex
    upload: UploadFile | None = None
    try:
        # 1) auth FIRST: the multipart body is never parsed for an unauthorized caller.
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
        limit = int(state.settings.max_upload_mb * 1024 * 1024)
        field_limit = min(_MULTIPART_FIELD_LIMIT, limit + _MULTIPART_OVERHEAD)
        # 2) Fast rejection ONLY when the declared length is absurdly large. A
        # small/absent/false Content-Length is never trusted: the bounded read in
        # _bounded_upload_bytes is the authoritative per-file check.
        raw_length = request.headers.get("content-length", "")
        if raw_length.isdigit() and int(raw_length) > limit + _MULTIPART_OVERHEAD:
            return _error(
                413,
                "file_too_large",
                f"The upload exceeds the {int(state.settings.max_upload_mb)} MB limit.",
                request_id,
            )
        # 3) Bounded strict multipart parse (text parts capped; at most one file,
        # at most four fields - anything else is rejected, never ignored).
        try:
            form = await request.form(max_files=1, max_fields=4, max_part_size=field_limit)
        except MultiPartException as exc:  # raised directly by some parser paths
            log.info("Bearing upload rejected by parser limits: %s", str(exc)[:120])
            return _error(
                422, "request_validation", "Malformed or oversized multipart form data.", request_id
            )
        except StarletteHTTPException as exc:  # starlette wraps parser-limit errors as HTTP 400
            log.info("Bearing upload rejected by parser limits: %s", str(exc.detail)[:120])
            return _error(
                422, "request_validation", "Malformed or oversized multipart form data.", request_id
            )
        except UnicodeDecodeError:
            return _error(422, "request_validation", "Multipart form data is not decodable.", request_id)
        items = form.multi_items()
        counts: dict[str, int] = {}
        for key, _value in items:
            counts[key] = counts.get(key, 0) + 1
        duplicates = sorted(key for key, n in counts.items() if n > 1)
        unknown = sorted(key for key in counts if key not in _FORM_FIELDS)
        if duplicates or unknown:
            return _error(
                422,
                "request_validation",
                "Form fields must be unique and within the allowed set "
                "(file, machine_id, machine_type, observed_at).",
                request_id,
                details={"duplicate": duplicates, "unknown": unknown},
            )
        parts = {key: value for key, value in items}
        candidate = parts.get("file")
        if not isinstance(candidate, UploadFile):
            return _error(
                422,
                "request_validation",
                "Exactly one uploaded audio file is required (form field 'file').",
                request_id,
                details={"param": "file"},
            )
        upload = candidate
        machine_id = parts.get("machine_id")
        machine_type = parts.get("machine_type")
        observed_raw = parts.get("observed_at")
        if not isinstance(machine_id, str) or not machine_id:
            return _error(
                422,
                "request_validation",
                "machine_id is required.",
                request_id,
                details={"param": "machine_id"},
            )
        if machine_type is not None and (not isinstance(machine_type, str) or not machine_type):
            return _error(
                422, "request_validation", "machine_type must be a non-empty string.", request_id,
                details={"param": "machine_type"},
            )
        if observed_raw is not None and not isinstance(observed_raw, str):
            return _error(
                422, "request_validation", "observed_at must be an ISO 8601 string.", request_id,
                details={"param": "observed_at"},
            )
        # 4) Observed timestamp: timezone-aware ISO only (naive rejected, never
        # fabricated); omitted stays None and is surfaced by TIMESTAMP_MISSING.
        observed_at = None
        if observed_raw:
            try:
                parsed_dt = datetime.fromisoformat(observed_raw.replace("Z", "+00:00"))
            except ValueError:
                parsed_dt = None
            if parsed_dt is None or parsed_dt.tzinfo is None:
                return _error(
                    422,
                    "request_validation",
                    "observed_at must be timezone-aware ISO 8601 (naive timestamps are rejected).",
                    request_id,
                    details={"param": "observed_at"},
                )
            observed_at = parsed_dt.astimezone(timezone.utc)
        # 5) Machine + declared-type consistency: machine_type is metadata that must
        # match the fleet record (or be rejected when the record cannot verify it).
        machine = state.fleet.machines.get(machine_id)
        if machine is None:
            return _error(
                422,
                "unknown_machine",
                "machine_id is not a machine in the configured demo fleet.",
                request_id,
                details={"machine_id": machine_id},
            )
        profile = machine.spec.acoustic_profile
        if machine_type is not None and machine_type != profile:
            return _error(
                422,
                "machine_type_mismatch",
                "machine_type does not match the selected machine's recorded type in the demo fleet "
                "(machines without a recorded type cannot verify one).",
                request_id,
                details={"machine_type": machine_type, "acoustic_profile": profile},
            )
        # 6) Exactly one model for this slice - a missing model is an explicit 503,
        # never a fallback to another model, never a dynamically enabled type.
        classifier = state.registry.require_bearing()
        identity = state.registry.bearing_identity
        reference_fp = (identity or {}).get("reference_fingerprint")
        # 7) Shaft speed from live telemetry (legacy /acoustic/diagnose machine_id
        # behaviour). No client rpm field exists in this contract; when the speed
        # is unknown the classifier's configured default is assumed and reported.
        rpm: float | None = None
        reading = machine.reading or {}
        if reading.get("rotational_speed_rpm") is not None:
            rpm = float(reading["rotational_speed_rpm"])
        # 8) Bounded read (authoritative size check) + resource cleanup on every path.
        # No extension allowlist and no MATLAB rejection: decode success is the rule
        # on the original bearing path (.mat is this dataset's native format).
        filename = _sanitize_filename(upload.filename)
        data = await _bounded_upload_bytes(upload, limit)
        upload = None  # closed by _bounded_upload_bytes
        # 9) Threadpool: optional header probe + finite/min-duration validation +
        # UNCHANGED scoring.
        result, original_s = await run_in_threadpool(_diagnose_buffer, classifier, io.BytesIO(data), rpm)
        output_doc = dict(result)
        output_doc["filename"] = filename
        # Typed output validated FIRST: evidence is read from the validated model,
        # so a malformed ML result is a server-side bug (500, nothing saved).
        output = FaultDiagnosis.model_validate(output_doc)
        processed_s = float(output.duration_s)
        cap = classifier.metadata.get("max_audio_seconds")
        truncated = original_s is not None and cap is not None and processed_s < original_s - 0.05
        extra_flags: list[QualityFlag] = []
        if truncated:
            note = (
                f"The recording is about {original_s:.1f} s but the classifier processes at most "
                f"{cap} s; only the first {processed_s:.1f} s were scored (scoring behaviour unchanged)."
            )
            extra_flags.append(QualityFlag(code="AUDIO_TRUNCATED", severity="info", explanation=note))
        # 10) Snapshot: server-owned everything; typed ML output preserved as-is
        # (filename added as sanitized untrusted display metadata). Evidence carries
        # what this classifier actually produces - class confidence and durations;
        # it has NO anomaly score and NO conformal p-value, so none is invented.
        evidence = [
            EvidenceDraft(
                kind="class_confidence",
                value=output.confidence,
                unit="probability",  # mean segment probability of the predicted class
                source="bearing",
                source_mode="uploaded",
                observed_at=observed_at,
            )
        ]
        if original_s is not None:
            evidence.append(
                EvidenceDraft(
                    kind="audio_duration_original_s",
                    value=original_s,
                    unit="seconds",  # header-derived original length; unavailable for .mat, never fabricated
                    source="bearing",
                    source_mode="uploaded",
                    observed_at=observed_at,
                )
            )
        evidence.append(
            EvidenceDraft(
                kind="audio_duration_processed_s",
                value=processed_s,
                unit="seconds",  # what the model actually scored
                source="bearing",
                source_mode="uploaded",
                observed_at=observed_at,
            )
        )
        identity_public = (
            {key: value for key, value in identity.items() if key in ModelIdentity.model_fields}
            if identity
            else None
        )
        snapshot = assemble_assessment(
            machine_id=machine_id,
            source_mode="uploaded",  # server-fixed: an upload is an upload
            expected_modalities=["bearing"],  # server-defined bearing-only profile
            modalities=[
                ModalityDraft(
                    kind="bearing",
                    availability="available",
                    reason_code="ok",
                    source_mode="uploaded",
                    observed_at=observed_at,
                    inference_at=datetime.now(timezone.utc),
                    model_identity=ModelIdentity.model_validate(identity_public) if identity_public else None,
                    recording_id=str(uuid4()),
                    recording_sha256=hashlib.sha256(data).hexdigest(),
                    reference_id=f"sha256:{reference_fp}" if reference_fp else None,
                    limitations=list(classifier.metadata.get("caveats", [])),
                    output=output,
                )
            ],
            evidence=evidence,
            extra_flags=extra_flags,
        )
        await run_in_threadpool(store.insert, snapshot, owner=actor)
        return Response(
            status_code=201,
            content=snapshot.model_dump_json(),
            media_type="application/json",
            headers={
                "Location": f"/api/v1/assessments/{snapshot.assessment_id}",
                "X-Request-ID": request_id,
            },
        )
    except _UploadProblem as exc:
        return _error(exc.status, exc.code, exc.message, request_id)
    except HTTPException as exc:  # require_bearing -> 503 with the static training hint
        code = "dependency_unavailable" if exc.status_code == 503 else "dependency_error"
        return _error(exc.status_code, code, str(exc.detail), request_id)
    except AssessmentAlreadyExists:
        return _error(409, "assessment_conflict", "The assessment already exists.", request_id)
    except Exception:
        log.exception("POST /assessments/bearing failed (request_id=%s); nothing was saved", request_id)
        return _error(500, "internal_error", "Unexpected server error; nothing was saved.", request_id)
    finally:
        if upload is not None:  # belt-and-braces: closed on every remaining path too
            try:
                await upload.close()
            except Exception:
                pass


@router.post("/multimodal", status_code=201)
async def create_multimodal_assessment(request: Request, state: AppState = Depends(get_state)) -> Response:
    """Create ONE assessment from at least two distinct modalities in a single request.

    Strict multipart contract: exactly ``machine_id``, ``sensor_json``,
    ``acoustic_file``, ``bearing_file`` and optional ``observed_at``; unknown,
    duplicate and malformed parts are rejected, never ignored, and scores,
    statuses, ids, references, model identities, owner, source_mode and
    expected_modalities are never accepted from form fields (``sensor_json`` is
    parsed with the strict legacy input contract: reading + demo source_mode
    only). At least TWO distinct modalities are required
    (``too_few_modalities``). Each modality has its own part name, so one part
    is never accepted as both acoustic and bearing evidence; separate modalities
    require separate parts with their own server evidence records (identical raw
    bytes in two DISTINCT parts are allowed and simply produce two records).

    Compatibility matrix, validated COMPLETELY before any inference (an
    unsupported requested combination fails the whole request, never partially):
    sensor needs a known machine and the loaded sensor model; acoustic needs the
    machine's non-null ``acoustic_profile`` plus the loaded model MATCHING that
    declared profile (this contract has no client-declared machine_type, so the
    legacy declared-type consistency check has nothing to verify) and refuses an
    actively enrolled reference; bearing runs the one deployed bearing
    classifier for ANY fleet machine - this slice is deliberately
    machine-type-independent for bearing (documented limitation: bearing input
    is selected explicitly through ``bearing_file`` and is NEVER inferred from a
    profile or a machine type).

    Provenance: the assessment ``source_mode`` is ``uploaded`` whenever any
    upload is present (always true on success here: two modalities need at least
    one upload), otherwise it falls back to the sensor-declared legacy demo mode.
    Per-modality source modes stay separate - the sensor keeps its declared
    ``dataset_demo``/``simulated`` mode - so mixed provenance is flagged by the
    EXISTING MIXED_SOURCE_PROVENANCE rule; no new source_mode literal exists.
    Retention is HASH-ONLY for both recordings: each uploaded part gets its own
    server UUID and sha256 of its raw bytes, the sanitized filename is display
    metadata only, and no audio file is ever written (the hash is an identity
    record, not replay capability).

    Snapshot: server-owned id/owner/timestamps/ids/hashes/identities; every
    modality keeps its OWN original typed output (no probability is fused across
    modalities - the overall status is the existing conservative assess/v1
    max-rank rule over the preserved per-modality outputs). ``expected_modalities``
    is exactly the server-validated requested set, so requested-multimodal
    completeness is 1.0 by construction - it says NOTHING about machine health
    or full machine coverage (deliberate scope vs missing evidence are never
    conflated).

    Upload safety: auth is validated BEFORE the multipart body is parsed; the
    parser is bounded (max_files=2, max_fields=4, text parts capped); every file
    is re-checked with a bounded read (Content-Length is only an absurd-length
    fast reject, never trusted when small/absent/false); upload resources are
    closed on every success and error path. DEPLOYMENT-LEVEL REQUEST LIMITS (not
    enforceable in-app): uvicorn applies no total request-body cap and the
    multipart parser spools file parts to the process TEMP directory while
    parsing - deployments accepting untrusted traffic must bound request size at
    the reverse proxy/host and keep TEMP on a writable, quota'd volume (D: in
    this deployment).

    Flow: auth -> bounded strict validation -> compatibility matrix -> threadpool
    inference for ALL requested modalities -> immutable snapshot -> single
    transactional insert -> 201 + Location. Any validation, inference or insert
    failure persists NO assessment and NO partial document.
    """
    request_id = uuid4().hex
    acoustic_upload: UploadFile | None = None
    bearing_upload: UploadFile | None = None
    try:
        # 1) auth FIRST: the multipart body is never parsed for an unauthorized caller.
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
        limit = int(state.settings.max_upload_mb * 1024 * 1024)
        field_limit = min(_MULTIPART_FIELD_LIMIT, limit + _MULTIPART_OVERHEAD)
        # 2) Fast rejection ONLY when the declared length is absurd for TWO files
        # plus the capped text parts. A small/absent/false Content-Length is never
        # trusted: the bounded per-file read below is the authoritative check.
        raw_length = request.headers.get("content-length", "")
        max_body = 2 * limit + _MULTIMODAL_MAX_FIELDS * field_limit + _MULTIPART_OVERHEAD
        if raw_length.isdigit() and int(raw_length) > max_body:
            return _error(
                413,
                "file_too_large",
                f"The upload exceeds the {int(state.settings.max_upload_mb)} MB per-file limit.",
                request_id,
            )
        # 3) Bounded strict multipart parse (text parts capped; at most two files,
        # at most four text fields - anything else is rejected, never ignored).
        try:
            form = await request.form(
                max_files=_MULTIMODAL_MAX_FILES, max_fields=_MULTIMODAL_MAX_FIELDS, max_part_size=field_limit
            )
        except MultiPartException as exc:  # raised directly by some parser paths
            log.info("Multimodal upload rejected by parser limits: %s", str(exc)[:120])
            return _error(422, "request_validation", "Malformed or oversized multipart form data.", request_id)
        except StarletteHTTPException as exc:  # starlette wraps parser-limit errors as HTTP 400
            log.info("Multimodal upload rejected by parser limits: %s", str(exc.detail)[:120])
            return _error(422, "request_validation", "Malformed or oversized multipart form data.", request_id)
        except UnicodeDecodeError:
            return _error(422, "request_validation", "Multipart form data is not decodable.", request_id)
        items = form.multi_items()
        counts: dict[str, int] = {}
        for key, _value in items:
            counts[key] = counts.get(key, 0) + 1
        duplicates = sorted(key for key, n in counts.items() if n > 1)
        unknown = sorted(key for key in counts if key not in _MULTIMODAL_FORM_FIELDS)
        if duplicates or unknown:
            return _error(
                422,
                "request_validation",
                "Form fields must be unique and within the allowed set "
                "(machine_id, sensor_json, acoustic_file, bearing_file, observed_at).",
                request_id,
                details={"duplicate": duplicates, "unknown": unknown},
            )
        parts = {key: value for key, value in items}  # duplicates rejected above: one entry per part
        # 4) Part discipline. Upload references are captured NOW so the finally
        # block closes them on every later validation/matrix/inference error path.
        machine_id = parts.get("machine_id")
        if not isinstance(machine_id, str) or not machine_id:
            return _error(
                422, "request_validation", "machine_id is required.", request_id, details={"param": "machine_id"}
            )
        sensor_raw = parts.get("sensor_json")
        if sensor_raw is not None and not isinstance(sensor_raw, str):
            return _error(
                422,
                "request_validation",
                "sensor_json must be a JSON form field, not a file.",
                request_id,
                details={"param": "sensor_json"},
            )
        acoustic_candidate = parts.get("acoustic_file")
        if acoustic_candidate is not None and not isinstance(acoustic_candidate, UploadFile):
            return _error(
                422,
                "request_validation",
                "acoustic_file must be an uploaded file.",
                request_id,
                details={"param": "acoustic_file"},
            )
        bearing_candidate = parts.get("bearing_file")
        if bearing_candidate is not None and not isinstance(bearing_candidate, UploadFile):
            return _error(
                422,
                "request_validation",
                "bearing_file must be an uploaded file.",
                request_id,
                details={"param": "bearing_file"},
            )
        acoustic_upload = acoustic_candidate
        bearing_upload = bearing_candidate
        # 5) At least TWO distinct modalities. Each modality maps to at most one
        # part name, so a single upload is never accepted as both acoustic and
        # bearing evidence (bearing is never inferred from acoustic_file, and
        # vice versa).
        requested = [
            kind
            for kind, present in (
                ("sensor", sensor_raw is not None),
                ("acoustic", acoustic_upload is not None),
                ("bearing", bearing_upload is not None),
            )
            if present
        ]
        if len(requested) < 2:
            return _error(
                422,
                "too_few_modalities",
                "At least two distinct modalities are required in one request "
                "(sensor_json, acoustic_file, bearing_file).",
                request_id,
                details={"present": requested},
            )
        # 6) Observed timestamp: timezone-aware ISO only (naive rejected, never
        # fabricated); omitted stays None and is surfaced by TIMESTAMP_MISSING.
        observed_raw = parts.get("observed_at")
        if observed_raw is not None and not isinstance(observed_raw, str):
            return _error(
                422,
                "request_validation",
                "observed_at must be an ISO 8601 string.",
                request_id,
                details={"param": "observed_at"},
            )
        observed_at = None
        if observed_raw:
            try:
                parsed_dt = datetime.fromisoformat(observed_raw.replace("Z", "+00:00"))
            except ValueError:
                parsed_dt = None
            if parsed_dt is None or parsed_dt.tzinfo is None:
                return _error(
                    422,
                    "request_validation",
                    "observed_at must be a timezone-aware ISO 8601 timestamp (naive timestamps are rejected).",
                    request_id,
                    details={"param": "observed_at"},
                )
            observed_at = parsed_dt.astimezone(timezone.utc)
        # 7) sensor_json: strict legacy input contract - inputs + demo provenance
        # only. Forged fields inside the JSON are rejected (extra="forbid"),
        # JSON syntax errors are their own sanitized reason, and a client can
        # never forge uploaded/live source_mode through this field.
        sensor_submission: SensorSubmission | None = None
        if sensor_raw is not None:
            try:
                parsed_json: Any = json.loads(sensor_raw)
            except json.JSONDecodeError:
                return _error(
                    422,
                    "request_validation",
                    "sensor_json must be a JSON object describing the sensor reading.",
                    request_id,
                    details={"param": "sensor_json", "reason": "invalid_json"},
                )
            try:
                sensor_submission = SensorSubmission.model_validate(parsed_json)
            except ValidationError as exc:
                return _error(
                    422,
                    "request_validation",
                    "sensor_json failed validation.",
                    request_id,
                    details=_validation_details(exc),
                )
        # 8) Machine association: an assessment is always machine-scoped.
        machine = state.fleet.machines.get(machine_id)
        if machine is None:
            return _error(
                422,
                "unknown_machine",
                "machine_id is not a machine in the configured demo fleet.",
                request_id,
                details={"machine_id": machine_id},
            )
        # 9) Bounded reads (authoritative per-file size/empty checks; Content-Length
        # is never trusted) plus the acoustic content allowlist, closed on every path.
        filename_a: str | None = None
        acoustic_bytes: bytes | None = None
        if acoustic_upload is not None:
            filename_a = _sanitize_filename(acoustic_upload.filename)
            acoustic_bytes = await _bounded_upload_bytes(acoustic_upload, limit)
            acoustic_upload = None  # closed by _bounded_upload_bytes
            if not filename_a or Path(filename_a).suffix.lower() not in AUDIO_UPLOAD_EXTENSIONS:
                return _error(
                    422, "invalid_audio", "Unsupported audio format; upload a WAV, FLAC or OGG file.", request_id
                )
            if acoustic_bytes[: len(MAT_MAGIC)] == MAT_MAGIC:
                return _error(
                    422,
                    "invalid_audio",
                    "MATLAB recordings are not accepted for acoustic evidence; upload WAV, FLAC or OGG.",
                    request_id,
                )
        filename_b: str | None = None
        bearing_bytes: bytes | None = None
        if bearing_upload is not None:
            filename_b = _sanitize_filename(bearing_upload.filename)
            bearing_bytes = await _bounded_upload_bytes(bearing_upload, limit)
            bearing_upload = None  # closed by _bounded_upload_bytes
            # No extension allowlist and no MATLAB rejection: decode success is the
            # rule on the original bearing path (.mat is that dataset's native format).
        # 10) Compatibility matrix: the COMPLETE requested set is validated before
        # any inference runs. Every rule below uses an existing legacy code (or the
        # legacy require_* 503/404 hint); unsupported_modality_combination is the
        # reserved catch-all for a matrix violation WITHOUT a legacy code.
        predictor = None
        if "sensor" in requested:
            predictor = state.registry.require_sensor()  # raises HTTPException(503, hint)
        detector = None
        acoustic_identity: dict[str, Any] | None = None
        if "acoustic" in requested:
            profile = machine.spec.acoustic_profile
            if not profile:
                return _error(
                    422,
                    "acoustic_profile_mismatch",
                    "The selected machine has no acoustic profile; acoustic assessments need one.",
                    request_id,
                    details={"machine_id": machine_id},
                )
            # The model must MATCH the machine's declared profile; a missing model
            # for that profile is the legacy require_* error, never a fallback to
            # another type and never a dynamically enabled profile.
            detector = state.registry.require_acoustic(profile)
            # Mutable enrolled references can never be identified/snapshotted safely
            # here: refuse instead of recording a bare condition name as provenance.
            if getattr(detector, "enrolled", None):
                return _error(
                    409,
                    "enrolled_reference_active",
                    "This sound model has enrolled conditions; the exact enrolled reference state cannot be "
                    "identified and snapshotted safely in this slice, so no assessment was created.",
                    request_id,
                    details={"enrolled": sorted(getattr(detector, "enrolled", {}))},
                )
            acoustic_identity = state.registry.acoustic_identity.get(profile)
        classifier = None
        bearing_identity: dict[str, Any] | None = None
        if "bearing" in requested:
            classifier = state.registry.require_bearing()  # raises HTTPException(503, hint)
            bearing_identity = state.registry.bearing_identity
        if predictor is None and detector is None and classifier is None:
            # Unreachable with >= 2 validated modalities above; the contract
            # reserves this code for a matrix violation with no legacy code, so a
            # future rule can never silently fall through to a success path.
            return _error(
                422,
                "unsupported_modality_combination",
                "The requested modality combination is not supported by this endpoint.",
                request_id,
            )
        # 11) Threadpool inference for ALL requested modalities (fixed order; a
        # failure in any one aborts the whole request with nothing saved).
        inference_at = datetime.now(timezone.utc)
        sensor_result: dict[str, Any] | None = None
        if predictor is not None and sensor_submission is not None:
            [sensor_result] = await run_in_threadpool(predictor.assess, [sensor_submission.reading.model_dump()])
        acoustic_result: dict[str, Any] | None = None
        acoustic_original_s: float | None = None
        acoustic_processed_s: float | None = None
        acoustic_cap: float | None = None
        if detector is not None and acoustic_bytes is not None:
            acoustic_result, acoustic_original_s = await run_in_threadpool(
                _analyse_buffer, detector, io.BytesIO(acoustic_bytes)
            )
            acoustic_processed_s = float(acoustic_result["duration_s"])
            acoustic_cap = detector.metadata.get("max_audio_seconds")
        bearing_result: dict[str, Any] | None = None
        bearing_original_s: float | None = None
        bearing_processed_s: float | None = None
        bearing_cap: float | None = None
        bearing_rpm: float | None = None
        if classifier is not None and bearing_bytes is not None:
            # Shaft speed from live telemetry (legacy behaviour); never a client field.
            reading = machine.reading or {}
            if reading.get("rotational_speed_rpm") is not None:
                bearing_rpm = float(reading["rotational_speed_rpm"])
            bearing_result, bearing_original_s = await run_in_threadpool(
                _diagnose_buffer, classifier, io.BytesIO(bearing_bytes), bearing_rpm
            )
            bearing_processed_s = float(bearing_result["duration_s"])
            bearing_cap = classifier.metadata.get("max_audio_seconds")
        # 12) Immutable snapshot inputs: server-owned everything; each modality's
        # original typed output is preserved separately (+ sanitized display
        # filename for uploads). Typed output validation failure is a server-side
        # bug: it propagates to the generic handler -> 500, nothing saved.
        modalities: list[ModalityDraft] = []
        evidence: list[EvidenceDraft] = []
        if sensor_submission is not None and sensor_result is not None:
            identity_s = state.registry.sensor_identity
            modalities.append(
                ModalityDraft(
                    kind="sensor",
                    availability="available",
                    reason_code="ok",
                    source_mode=sensor_submission.source_mode,  # declared demo provenance, never forged
                    observed_at=observed_at,
                    inference_at=inference_at,
                    model_identity=ModelIdentity.model_validate(identity_s) if identity_s else None,
                    limitations=list(predictor.metadata.get("caveats", [])) if predictor else [],
                    output=SensorAssessment.model_validate(sensor_result),
                )
            )
            evidence.append(
                EvidenceDraft(
                    kind="failure_probability",
                    value=sensor_result["failure_probability"],
                    unit="probability",
                    source="sensor",
                    source_mode=sensor_submission.source_mode,
                    observed_at=observed_at,
                )
            )
        if acoustic_result is not None:
            output_doc = dict(acoustic_result)
            output_doc["filename"] = filename_a  # sanitized untrusted display metadata, like the legacy route
            identity_a_public = (
                {key: value for key, value in acoustic_identity.items() if key in ModelIdentity.model_fields}
                if acoustic_identity
                else None
            )
            modalities.append(
                ModalityDraft(
                    kind="acoustic",
                    availability="available",
                    reason_code="ok",
                    source_mode="uploaded",  # an upload is an upload
                    observed_at=observed_at,
                    inference_at=inference_at,
                    model_identity=ModelIdentity.model_validate(identity_a_public) if identity_a_public else None,
                    recording_id=str(uuid4()),  # one server UUID per uploaded part (1:1 evidence linkage)
                    recording_sha256=hashlib.sha256(acoustic_bytes).hexdigest() if acoustic_bytes else None,
                    reference_id=f"sha256:{acoustic_identity.get('reference_fingerprint')}"
                    if acoustic_identity and acoustic_identity.get("reference_fingerprint")
                    else None,
                    limitations=list(detector.metadata.get("caveats", [])) if detector else [],
                    output=AcousticAnalysis.model_validate(output_doc),
                )
            )
            evidence.append(
                EvidenceDraft(
                    kind="anomaly_score",
                    value=acoustic_result["score"],
                    unit="score",  # higher = more unusual than the normal reference
                    source="acoustic",
                    source_mode="uploaded",
                    observed_at=observed_at,
                )
            )
            if acoustic_result.get("p_value") is not None:
                evidence.append(
                    EvidenceDraft(
                        kind="conformal_p_value",
                        value=acoustic_result["p_value"],
                        unit=None,  # share of normal calibration recordings at least this unusual
                        source="acoustic",
                        source_mode="uploaded",
                        observed_at=observed_at,
                    )
                )
            if acoustic_original_s is not None:
                evidence.append(
                    EvidenceDraft(
                        kind="audio_duration_original_s",
                        value=acoustic_original_s,
                        unit="seconds",  # header-derived original length, before the processing cap
                        source="acoustic",
                        source_mode="uploaded",
                        observed_at=observed_at,
                    )
                )
            evidence.append(
                EvidenceDraft(
                    kind="audio_duration_processed_s",
                    value=acoustic_processed_s,
                    unit="seconds",  # what the model actually scored
                    source="acoustic",
                    source_mode="uploaded",
                    observed_at=observed_at,
                )
            )
        if bearing_result is not None:
            output_b = dict(bearing_result)
            output_b["filename"] = filename_b  # sanitized untrusted display metadata, like the legacy route
            bearing_output = FaultDiagnosis.model_validate(output_b)
            identity_b_public = (
                {key: value for key, value in bearing_identity.items() if key in ModelIdentity.model_fields}
                if bearing_identity
                else None
            )
            modalities.append(
                ModalityDraft(
                    kind="bearing",
                    availability="available",
                    reason_code="ok",
                    source_mode="uploaded",  # an upload is an upload
                    observed_at=observed_at,
                    inference_at=inference_at,
                    model_identity=ModelIdentity.model_validate(identity_b_public) if identity_b_public else None,
                    recording_id=str(uuid4()),  # one server UUID per uploaded part (1:1 evidence linkage)
                    recording_sha256=hashlib.sha256(bearing_bytes).hexdigest() if bearing_bytes else None,
                    reference_id=f"sha256:{bearing_identity.get('reference_fingerprint')}"
                    if bearing_identity and bearing_identity.get("reference_fingerprint")
                    else None,
                    limitations=list(classifier.metadata.get("caveats", [])) if classifier else [],
                    output=bearing_output,
                )
            )
            evidence.append(
                EvidenceDraft(
                    kind="class_confidence",
                    value=bearing_output.confidence,
                    unit="probability",  # mean segment probability of the predicted class
                    source="bearing",
                    source_mode="uploaded",
                    observed_at=observed_at,
                )
            )
            if bearing_original_s is not None:
                evidence.append(
                    EvidenceDraft(
                        kind="audio_duration_original_s",
                        value=bearing_original_s,
                        unit="seconds",  # header-derived; unavailable for .mat, never fabricated
                        source="bearing",
                        source_mode="uploaded",
                        observed_at=observed_at,
                    )
                )
            evidence.append(
                EvidenceDraft(
                    kind="audio_duration_processed_s",
                    value=bearing_processed_s,
                    unit="seconds",  # what the model actually scored
                    source="bearing",
                    source_mode="uploaded",
                    observed_at=observed_at,
                )
            )
        # Deterministic boundary fact (existing QualityCode, one flag even when
        # both uploads exceeded their model's processing cap): route-supplied flags
        # never change status computation.
        truncation_notes: list[str] = []
        if (
            acoustic_original_s is not None
            and acoustic_cap is not None
            and acoustic_processed_s is not None
            and acoustic_processed_s < acoustic_original_s - 0.05
        ):
            truncation_notes.append(
                f"the acoustic recording (~{acoustic_original_s:.1f} s, first {acoustic_processed_s:.1f} s "
                f"scored, cap {acoustic_cap} s)"
            )
        if (
            bearing_original_s is not None
            and bearing_cap is not None
            and bearing_processed_s is not None
            and bearing_processed_s < bearing_original_s - 0.05
        ):
            truncation_notes.append(
                f"the bearing recording (~{bearing_original_s:.1f} s, first {bearing_processed_s:.1f} s "
                f"scored, cap {bearing_cap} s)"
            )
        extra_flags: list[QualityFlag] = []
        if truncation_notes:
            extra_flags.append(
                QualityFlag(
                    code="AUDIO_TRUNCATED",
                    severity="info",
                    explanation="The model processing cap truncated scoring for "
                    + " and ".join(truncation_notes)
                    + "; scoring behaviour unchanged.",
                )
            )
        # Assessment-level provenance: any upload makes the whole assessment
        # "uploaded". The sensor-only fallback below is unreachable with >= 2
        # modalities (a sensor-only request is rejected as too_few_modalities),
        # but the rule is implemented exactly as specified rather than assumed.
        if acoustic_bytes is not None or bearing_bytes is not None:
            assessment_source = "uploaded"
        else:
            assessment_source = sensor_submission.source_mode if sensor_submission else "simulated"
        snapshot = assemble_assessment(
            machine_id=machine_id,
            source_mode=assessment_source,
            # Exactly the server-validated requested set (canonical kind order);
            # all of them produced available evidence -> completeness 1.0 for THIS
            # requested scope only, never machine-health or coverage completeness.
            expected_modalities=[kind for kind in ("sensor", "acoustic", "bearing") if kind in requested],
            modalities=modalities,
            evidence=evidence,
            extra_flags=extra_flags,
        )
        await run_in_threadpool(store.insert, snapshot, owner=actor)
        return Response(
            status_code=201,
            content=snapshot.model_dump_json(),
            media_type="application/json",
            headers={
                "Location": f"/api/v1/assessments/{snapshot.assessment_id}",
                "X-Request-ID": request_id,
            },
        )
    except _UploadProblem as exc:
        return _error(exc.status, exc.code, exc.message, request_id)
    except HTTPException as exc:  # require_sensor/require_acoustic/require_bearing -> 503/404 + static hint
        code = "dependency_unavailable" if exc.status_code == 503 else "dependency_error"
        return _error(exc.status_code, code, str(exc.detail), request_id)
    except AssessmentAlreadyExists:
        return _error(409, "assessment_conflict", "The assessment already exists.", request_id)
    except Exception:
        log.exception("POST /assessments/multimodal failed (request_id=%s); nothing was saved", request_id)
        return _error(500, "internal_error", "Unexpected server error; nothing was saved.", request_id)
    finally:
        # Belt-and-braces: any upload not yet owned by _bounded_upload_bytes is
        # closed here, so no path (success, validation error, matrix error,
        # inference error, insert error) leaks an upload resource.
        for pending in (acoustic_upload, bearing_upload):
            if pending is not None:
                try:
                    await pending.close()
                except Exception:
                    pass
