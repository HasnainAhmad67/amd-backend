"""AMD Token Factory inference provider (slice S3 + analyze wiring).

This module is the transport-only boundary between the Central AI Agent routes that
are approved for AMD (``/api/v1/ai/explain`` and ``/api/v1/ai/analyze``) and the
documented AMD Token Factory shared-endpoint chat API. It uses the already installed
``httpx`` client (no new SDK) and a static Bearer API key.

Documented contract (AMD Token Factory guide):
- ``POST {base}/chat/completions`` with base
  ``https://developer.amd.com.cn/radeon/api/v1``;
- ``Authorization: Bearer rc-...`` (static key; no OAuth in this slice);
- OpenAI-compatible response: ``choices[0].message.content``;
- ``response_format {"type":"json_object"}``, ``stream false``, ``temperature 0``;
- 429 carries ``Retry-After`` (per-minute 60 s, concurrency 1 s); retry only
  429/502/503/network/timeout;
- the provider keeps prompts/outputs for about two days and is a trial/development
  service with no SLA.

Boundary rules enforced here:
- the provider receives ONLY the facts-only projection of a snapshot (``request.facts``)
  - never the ``Assessment`` object, raw audio, filesystem paths, model fingerprints,
  SQL or store internals - and the request body is a minimized serializable payload;
- it returns an UNTRUSTED prose candidate (:class:`AmdExplainProse`); the route owns
  every structural field, id, status, score, timestamp and the persistence decision;
- the API key is never logged, echoed, serialized or embedded in an exception.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Callable, Mapping

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..core.config import Settings, is_allowed_amd_base_url
from ..schemas.ai import (
    MAX_EXPLANATION_CHARS,
    MAX_NOTE_CHARS,
    PROMPT_VERSION_ANALYZE,
    PROMPT_VERSION_EXPLAIN,
    PROMPT_VERSION_REVIEW,
)
from .ai_provider import (
    ProviderError,
    ProviderFailureClass,
    ProviderFeature,
    ProviderRequest,
    ProviderResult,
    ProviderUsage,
)

log = logging.getLogger("telltale.ai.amd")

CHAT_COMPLETIONS_PATH = "/chat/completions"

# Only these upstream status classes are retried (documented transient failures).
_RETRYABLE_STATUS_CLASSES = frozenset(
    {ProviderFailureClass.RATE_LIMITED, ProviderFailureClass.NETWORK}
)

# System prompt for the documented JSON-object response format. The exact phrases the
# existing grounding validator requires ("model indication", "confirmed fault",
# "missing") are requested explicitly so a compliant answer can validate unchanged.

JSON_INSTRUCTIONS = (
    "You explain an immutable predictive-maintenance assessment snapshot for a "
    "maintenance technician. Use ONLY the JSON facts provided. Never invent "
    "numbers, ids, timestamps, statuses, causes, parts or recommendations. "
    "Do not use double quotes and do not mention dates or times. "
    "Write a concise explanation using short paragraphs separated by newline "
    "characters. Organize it in this order: "
    "Assessment summary: explain the stored overall status in plain language. "
    "Available evidence: summarize only the supplied stored indications. "
    "Data limitations: explain missing evidence, quality flags and stored "
    "limitations when present. "
    "Do not claim that absent limitations mean the data is complete. "
    "Include the exact phrase \"model indications, not confirmed faults\" "
    "and state that missing evidence limits interpretation. "
    "Preserve uncertainty and stored limitations. Never claim machine-health "
    "completeness or coverage. Return ONLY a JSON object with the keys "
    '"explanation_text" (string) and optionally "uncertainty_notes" (array of strings).'
)


# System prompt for the documented JSON-object response format for the analyze
# feature. The exact phrases the existing analyze grounding validator requires
# ("not a fault diagnosis", "not machine-health completeness", "not coverage",
# "no fused failure probability") are requested explicitly so a compliant answer can
# validate unchanged. Co-occurrence only - never a combined/fused value or causality.
ANALYZE_JSON_INSTRUCTIONS = (
    "You describe how the stored indications of an immutable predictive-maintenance "
    "assessment relate across its available modalities for a maintenance technician. "
    "Use ONLY the JSON facts provided. Quote stored values exactly; never invent "
    "numbers, ids, timestamps, statuses, causes, parts or recommendations. Do not use "
    "double quotes and do not mention dates or times. Write 3-6 plain sentences "
    "describing the co-occurrence of the stored indications without combining them "
    "into one value, score or probability. Include the exact phrases \"not a fault "
    "diagnosis\", \"not machine-health completeness\", \"not coverage\" and \"no fused "
    "failure probability\". Never state that one modality explains another and never "
    "claim causality. Preserve uncertainty and the stored limitations. Return ONLY a "
    'JSON object with the key "analysis_text" (string).'
)

# System prompt for the documented JSON-object response format for the evidence-review
# feature. The exact phrases the existing review grounding validator requires
# ("not a fault diagnosis", "not machine-health completeness", "not coverage") plus the
# literal word "requested" are requested explicitly so a compliant answer can validate
# unchanged. Stored evidence and uncertainty only - never a fault claim or a score.
REVIEW_JSON_INSTRUCTIONS = (
    "You review the stored evidence and uncertainty of an immutable predictive-maintenance "
    "assessment snapshot for a maintenance technician. Use ONLY the JSON facts provided. "
    "Quote stored values exactly; never invent numbers, ids, timestamps, statuses, causes, "
    "parts or recommendations. Do not use double quotes and do not mention dates or times. "
    "Write 3-6 plain sentences describing the stored evidence, the stored quality flags and "
    "the stored limitations. Include the exact phrases \"not a fault diagnosis\", "
    "\"not machine-health completeness\" and \"not coverage\", and include the word "
    "\"requested\". Never claim a fault, a score, a p-value, a confidence or a fused "
    "probability, and never state that one modality explains another. Preserve uncertainty "
    "and the stored limitations. Return ONLY a JSON object with the key \"review_text\" "
    "(string)."
)


class AmdExplainProse(BaseModel):
    """The ONLY fields AMD is asked for: prose plus optional uncertainty notes.

    Everything else about an ``AiExplainOutput`` is set independently by the backend
    from the immutable snapshot. ``extra="forbid"`` rejects any invented extra field.
    """

    model_config = ConfigDict(extra="forbid")

    explanation_text: str = Field(min_length=120, max_length=MAX_EXPLANATION_CHARS)
    uncertainty_notes: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("uncertainty_notes")
    @classmethod
    def _bound_notes(cls, notes: list[str]) -> list[str]:
        bounded: list[str] = []
        for note in notes:
            flat = " ".join(str(note).split())[:MAX_NOTE_CHARS]
            if flat and flat not in bounded:
                bounded.append(flat)
        return bounded[:12]


class AmdAnalyzeProse(BaseModel):
    """The ONLY field AMD is asked for on analyze: the co-occurrence prose.

    Every structural field of an ``AiAnalyzeOutput`` (identity, modality summaries,
    pairwise relations, agreement, provenance, non-claims, references) is set
    independently by the backend from the immutable snapshot, exactly like explain.
    ``extra="forbid"`` rejects any invented extra field.
    """

    model_config = ConfigDict(extra="forbid")

    analysis_text: str = Field(min_length=120, max_length=MAX_EXPLANATION_CHARS)


class AmdReviewProse(BaseModel):
    """The ONLY field AMD is asked for on evidence review: the review prose.

    Every structural field of an ``AiEvidenceReviewOutput`` (identity, modality reviews,
    quality flags, uncertainty indicators, provenance, non-claims, references) is set
    independently by the backend from the immutable snapshot, exactly like explain and
    analyze. ``extra="forbid"`` rejects any invented extra field.
    """

    model_config = ConfigDict(extra="forbid")

    review_text: str = Field(min_length=120, max_length=MAX_EXPLANATION_CHARS)


def minimize_for_amd(facts: Mapping[str, Any]) -> dict[str, Any]:
    """The minimized, serializable facts-only projection sent to AMD.

    Includes only: machine_id, stored overall status + status reasons, requested /
    present / missing modalities, availability reason codes, quality flag codes and
    severities, relevant evidence kinds and values, and stored limitations. Excludes
    local paths, secrets, raw audio, fingerprints, internal storage metadata,
    unnecessary ids (assessment/evidence ids) and any question echo.
    """
    modalities = list(facts.get("modalities") or [])
    present = [m.get("kind") for m in modalities if m.get("availability") == "available"]
    expected = list(facts.get("expected_modalities") or [])
    missing = [kind for kind in expected if kind not in present]
    return {
        "machine_id": facts.get("machine_id"),
        "overall_status": facts.get("overall_status"),
        "status_reasons": list(facts.get("status_reasons") or [])[:20],
        "expected_modalities": expected,
        "present_modalities": present,
        "missing_modalities": missing,
        "quality_flags": [
            {"code": f.get("code"), "severity": f.get("severity")}
            for f in (facts.get("quality_flags") or [])
        ],
        "modalities": [
            {
                "kind": m.get("kind"),
                "availability": m.get("availability"),
                "reason_code": m.get("reason_code"),
                "evidence": [
                    {"kind": e.get("kind"), "value": e.get("value"), "unit": e.get("unit")}
                    for e in (m.get("evidence") or [])
                ],
                "limitations": list(m.get("limitations") or [])[:20],
            }
            for m in modalities
        ],
    }


def _classify_status(status_code: int) -> ProviderFailureClass:
    if status_code == 429:
        return ProviderFailureClass.RATE_LIMITED
    if status_code in (502, 503):
        return ProviderFailureClass.NETWORK
    if status_code in (401, 403):
        return ProviderFailureClass.AUTH
    return ProviderFailureClass.PROVIDER_ERROR


def _status_message(status_code: int) -> str:
    # A bounded, sanitized classification - never the raw provider error body.
    return f"AMD responded with HTTP {status_code}"


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        value = float(raw.strip())
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _usage_from(node: Any) -> ProviderUsage:
    if not isinstance(node, dict):
        return ProviderUsage()

    def _int(value: Any) -> int | None:
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    return ProviderUsage(
        prompt_tokens=_int(node.get("prompt_tokens")),
        completion_tokens=_int(node.get("completion_tokens")),
        total_tokens=_int(node.get("total_tokens")),
    )


def _parse_content(content: str, *, model: type[BaseModel], label: str) -> BaseModel:
    """Strict parse of ``choices[0].message.content`` into the feature's prose schema.

    Accepts an optional ```json fence. Failures are reported with a bounded,
    sanitized message - the raw content is never included.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProviderError(
            ProviderFailureClass.MALFORMED, "AMD content is not a JSON object"
        ) from exc
    if not isinstance(obj, dict):
        raise ProviderError(
            ProviderFailureClass.MALFORMED, "AMD content is not a JSON object"
        )
    try:
        return model.model_validate(obj)
    except ValidationError as exc:
        raise ProviderError(
            ProviderFailureClass.MALFORMED, f"AMD content failed the {label} output schema"
        ) from exc


class _AmdChatProvider:
    """Static-key AMD chat provider base (no OAuth, no tools).

    Subclasses supply the feature discriminator, the system instructions, the prompt
    version and the strict prose schema for their feature. Everything else (auth,
    request shape, bounded retries, TLS/redirect policy, response parsing) is shared
    and identical for every AMD feature.
    """

    feature: ProviderFeature
    _instructions: str
    _prompt_version: str
    _prose_model: type[BaseModel]
    _schema_label: str

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        key = settings.amd_api_key.get_secret_value() if settings.amd_api_key else ""
        model = (settings.amd_model or "").strip()
        base = (settings.amd_api_base_url or "").rstrip("/")
        # Fail closed without echoing any secret value.
        if not key.strip() or not model:
            raise ProviderError(
                ProviderFailureClass.PROVIDER_ERROR,
                "AMD provider requires TELLTALE_AMD_API_KEY and TELLTALE_AMD_MODEL",
            )
        if not is_allowed_amd_base_url(base):
            raise ProviderError(
                ProviderFailureClass.PROVIDER_ERROR,
                "AMD provider base URL is not an allowed AMD https host/base path",
            )
        self._settings = settings
        self._key = key
        self._model = model
        self._url = base + CHAT_COMPLETIONS_PATH
        self._transport = transport
        self._sleep = sleep or time.sleep
        self._monotonic = monotonic or time.monotonic

    # -- identity ----------------------------------------------------------------
    @property
    def name(self) -> str:
        return "amd"

    @property
    def model(self) -> str | None:
        return self._model

    # -- request construction ----------------------------------------------------
    def build_request(self, context: Any, **_kwargs: Any) -> ProviderRequest:
        # The request carries the snapshot in the provider-neutral envelope, but the AI
        # provider reads ONLY the facts-only projection (``request.facts``); the raw
        # Assessment is never serialized or sent.
        return ProviderRequest(feature=self.feature, context=tuple(context))

    def prompt_sha256(self, request: ProviderRequest) -> str:
        """Deterministic digest of the effective AMD prompt (version + instructions +
        minimized facts). The raw prompt is never stored."""
        canonical = json.dumps(
            {
                "prompt_version": self._prompt_version,
                "instructions": self._instructions,
                "facts": minimize_for_amd(request.facts[0]) if request.facts else {},
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # -- generation --------------------------------------------------------------
    def generate(self, request: ProviderRequest) -> ProviderResult:
        if request.feature is not self.feature:
            raise ProviderError(
                ProviderFailureClass.PROVIDER_ERROR,
                f"AMD provider only implements the {self.feature.value} feature",
            )
        if not request.facts:
            raise ProviderError(
                ProviderFailureClass.PROVIDER_ERROR,
                "AMD provider requires a facts-only context",
            )
        body = self._build_body(minimize_for_amd(request.facts[0]))
        started = self._monotonic()
        deadline = started + float(self._settings.ai_timeout_s)
        content, usage = self._post_with_retries(body, deadline)
        prose = self._parse_content(content)
        usage = ProviderUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=(self._monotonic() - started) * 1000.0,
        )
        return ProviderResult(
            candidate=prose, usage=usage, provider="amd", model=self._model, raw_text=content
        )

    def _parse_content(self, content: str) -> BaseModel:
        return _parse_content(content, model=self._prose_model, label=self._schema_label)

    # -- transport ---------------------------------------------------------------
    def _build_body(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._instructions},
                {
                    "role": "user",
                    "content": json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        # ``max_tokens`` is OPTIONAL: some AMD models reject an explicit value with HTTP 400
        # (DeepSeek-V4-Flash reports its allowed maximum as null). Only include it when a
        # positive integer was configured - never a guessed model maximum, never null.
        if self._settings.ai_max_output_tokens is not None:
            body["max_tokens"] = int(self._settings.ai_max_output_tokens)
        return body

    def _retry_wait(
        self, attempt: int, retry_after: float | None, cap: float, deadline: float
    ) -> float | None:
        """Bounded wait before a retry, or None when there is no room / no budget.

        A long ``Retry-After`` beyond the configured cap never becomes an in-process
        wait: the caller fails over to the deterministic fallback immediately.
        """
        if retry_after is not None:
            if retry_after > cap:
                return None
            wait = retry_after
        else:
            wait = min(cap, 0.5 * (2 ** attempt))
        if self._monotonic() + wait >= deadline:
            return None
        return wait

    def _post_with_retries(
        self, body: Mapping[str, Any], deadline: float
    ) -> tuple[str, ProviderUsage]:
        max_retries = int(self._settings.ai_max_retries)
        cap = float(self._settings.ai_retry_max_s)
        timeout = httpx.Timeout(
            connect=5.0,
            read=float(self._settings.ai_timeout_s),
            write=10.0,
            pool=5.0,
        )
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        attempt = 0
        with httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            transport=self._transport,
        ) as client:
            while True:
                if self._monotonic() >= deadline:
                    raise ProviderError(
                        ProviderFailureClass.TIMEOUT,
                        "AMD request exceeded the bounded deadline",
                    )
                try:
                    response = client.post(self._url, json=body, headers=headers)
                except httpx.TimeoutException as exc:
                    wait = self._retry_wait(attempt, None, cap, deadline) if attempt < max_retries else None
                    if wait is None:
                        raise ProviderError(
                            ProviderFailureClass.TIMEOUT, "AMD request timed out"
                        ) from exc
                    attempt += 1
                    self._sleep(wait)
                    continue
                except httpx.TransportError as exc:
                    wait = self._retry_wait(attempt, None, cap, deadline) if attempt < max_retries else None
                    if wait is None:
                        raise ProviderError(
                            ProviderFailureClass.NETWORK, "AMD network error"
                        ) from exc
                    attempt += 1
                    self._sleep(wait)
                    continue
                except httpx.HTTPError as exc:
                    raise ProviderError(
                        ProviderFailureClass.PROVIDER_ERROR, "AMD transport error"
                    ) from exc

                if response.status_code == 200:
                    return self._parse_ok(response)

                failure = _classify_status(response.status_code)
                if failure in _RETRYABLE_STATUS_CLASSES and attempt < max_retries:
                    wait = self._retry_wait(
                        attempt, _retry_after_seconds(response), cap, deadline
                    )
                    if wait is not None:
                        attempt += 1
                        self._sleep(wait)
                        continue
                raise ProviderError(failure, _status_message(response.status_code))

    def _parse_ok(self, response: httpx.Response) -> tuple[str, ProviderUsage]:
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderError(
                ProviderFailureClass.MALFORMED, "AMD response was not JSON"
            ) from exc
        if not isinstance(data, dict):
            raise ProviderError(ProviderFailureClass.MALFORMED, "AMD response was not a JSON object")
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                ProviderFailureClass.MALFORMED,
                "AMD response is missing choices[0].message.content",
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError(
                ProviderFailureClass.MALFORMED, "AMD response content is empty"
            )
        return content, _usage_from(data.get("usage"))


class AmdExplainProvider(_AmdChatProvider):
    """Static-key AMD chat provider for the explain feature (no OAuth, no tools)."""

    feature = ProviderFeature.EXPLAIN
    _instructions = JSON_INSTRUCTIONS
    _prompt_version = PROMPT_VERSION_EXPLAIN
    _prose_model = AmdExplainProse
    _schema_label = "explain"


class AmdAnalyzeProvider(_AmdChatProvider):
    """Static-key AMD chat provider for the analyze feature (co-occurrence prose only).

    Same transport, auth and bounded-retry policy as explain; only the system
    instructions, the prompt version and the strict prose schema differ. It never
    receives raw audio, paths, ids, fingerprints or secrets - only the minimized
    facts-only projection - and it returns an untrusted prose candidate that the
    unchanged analyze grounding validator must accept before anything is persisted.
    """

    feature = ProviderFeature.ANALYZE
    _instructions = ANALYZE_JSON_INSTRUCTIONS
    _prompt_version = PROMPT_VERSION_ANALYZE
    _prose_model = AmdAnalyzeProse
    _schema_label = "analyze"


class AmdReviewProvider(_AmdChatProvider):
    """Static-key AMD chat provider for the evidence-review feature (prose only).

    Same transport, auth and bounded-retry policy as explain/analyze; only the system
    instructions, the prompt version and the strict prose schema differ. It returns an
    untrusted prose candidate that the unchanged evidence-review grounding validator must
    accept before anything is persisted.
    """

    feature = ProviderFeature.EVIDENCE_REVIEW
    _instructions = REVIEW_JSON_INSTRUCTIONS
    _prompt_version = PROMPT_VERSION_REVIEW
    _prose_model = AmdReviewProse
    _schema_label = "evidence_review"


__all__ = [
    "ANALYZE_JSON_INSTRUCTIONS",
    "AmdAnalyzeProse",
    "AmdAnalyzeProvider",
    "AmdExplainProse",
    "AmdExplainProvider",
    "AmdReviewProse",
    "AmdReviewProvider",
    "JSON_INSTRUCTIONS",
    "REVIEW_JSON_INSTRUCTIONS",
    "minimize_for_amd",
]
