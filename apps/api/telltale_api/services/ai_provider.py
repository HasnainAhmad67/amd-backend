"""Provider-neutral boundary for the five Central AI Agent features (slice S1 + S2).

This module introduces an INTERNAL abstraction between the AI routes and the code
that generates a candidate answer. It does NOT add any new provider: the only
implementation is the existing deterministic, offline template provider, exposed
through :class:`TemplateAIProvider` and the :func:`get_ai_provider` factory.

Slice S2 adds configuration-driven SELECTION: the factory reads the provider name
from the application settings and fails closed for any provider that is not
implemented (never a silent downgrade to template). No client, credentials, token
issuance, network access, quota or audit live in THIS module.

Slice S3 adds one more selectable provider - the AMD Token Factory static-key client
for the explain, analyze and evidence-review features - implemented in the separate
``ai_amd_provider`` module and imported lazily by the factory. This module stays free of
any transport/httpx dependency: it only routes to a provider by name.

Contract (what a provider is allowed to be):

- A provider receives a :class:`ProviderRequest` - the immutable, owner-visible
  assessment context (facts source) plus the feature discriminator and the
  feature-specific request metadata. The facts-only projection of that context is
  available as :attr:`ProviderRequest.facts`.
- A provider NEVER receives assessment stores, the model registry, ML models, raw
  audio, fleet telemetry, filesystem paths or secrets. Settings are read ONLY to pick
  the provider name - no credential or secret is ever reachable from a request.
- A provider NEVER mutates assessment snapshots or AI output records, and NEVER
  produces final validated evidence. It returns a raw/text-or-structured
  :class:`ProviderResult` (a candidate plus usage metadata) only.
- Evidence-grounding validation stays entirely OUTSIDE the provider: the routes
  parse/validate the candidate with the existing grounding validators before any
  persistence, exactly as today.

No behavior change is intended: the template provider is delegated to unchanged,
so outputs, prompt digests, validation statuses and persisted records are
byte-identical to the pre-slice code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

from ..core.config import (
    PLANNED_AI_PROVIDERS,
    SUPPORTED_AI_PROVIDERS,
    Settings,
    get_settings,
)
from ..schemas.assessment import Assessment
from .ai_analyze_template import TemplateAnalyzeProvider
from .ai_ask_template import TemplateAskProvider
from .ai_report_template import TemplateReportProvider
from .ai_review_template import TemplateEvidenceReviewProvider
from .ai_template import TemplateExplainProvider, project_assessment


class ProviderFeature(str, Enum):
    """The five generator features (values match the stored ``feature``/``kind``)."""

    EXPLAIN = "explain"
    EVIDENCE_REVIEW = "evidence_review"
    ANALYZE = "analyze"
    REPORT = "report"
    ASK = "ask"


class ProviderFailureClass(str, Enum):
    """Classification of a provider failure (used for routing/fallback decisions)."""

    TIMEOUT = "timeout"
    NETWORK = "network"
    CANCELLED = "cancelled"
    AUTH = "auth"
    RATE_LIMITED = "rate_limited"
    PROVIDER_ERROR = "provider_error"
    MALFORMED = "malformed"
    PROVIDER_REFUSAL = "provider_refusal"


# Failure classes that are safe to retry (transient); everything else must not be
# retried blindly (auth/validation/refusal/malformed are deterministic).
_RETRYABLE_FAILURES = frozenset(
    {
        ProviderFailureClass.TIMEOUT,
        ProviderFailureClass.NETWORK,
        ProviderFailureClass.RATE_LIMITED,
    }
)


class ProviderError(Exception):
    """A classified provider failure (never a grounding/validation failure).

    Grounding validation is NOT a provider concern: providers raise this only for
    their own transport/generation failures, and every failure path in the routes
    falls back or refuses rather than persisting invalid output as validated.
    """

    def __init__(
        self,
        failure_class: ProviderFailureClass,
        message: str,
        *,
        retryable: bool | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.message = message
        self.retryable = (
            failure_class in _RETRYABLE_FAILURES if retryable is None else bool(retryable)
        )
        self.detail = detail

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ProviderError(failure_class={self.failure_class.value!r}, "
            f"retryable={self.retryable})"
        )


class ProviderTimeout(ProviderError):
    """The provider did not complete within the caller's deadline."""

    def __init__(self, message: str = "provider timed out", **kwargs: Any) -> None:
        super().__init__(ProviderFailureClass.TIMEOUT, message, **kwargs)


class ProviderCancelled(ProviderError):
    """The provider call was cancelled by the caller (never retried)."""

    def __init__(self, message: str = "provider call cancelled", **kwargs: Any) -> None:
        super().__init__(
            ProviderFailureClass.CANCELLED, message, retryable=False, **kwargs
        )


@dataclass(frozen=True)
class ProviderUsage:
    """Usage metadata returned with a candidate (unknown values stay ``None``)."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None


@dataclass(frozen=True)
class ProviderRequest:
    """Facts-only context for one generation call plus the feature discriminator.

    ``context`` holds the immutable, owner-visible snapshots (the facts source);
    :attr:`facts` exposes their facts-only projection. The remaining fields are the
    feature-specific request metadata already resolved by the route (never
    user-supplied identifiers beyond the bounded sanitized question).
    """

    feature: ProviderFeature
    context: tuple[Assessment, ...] = ()
    question: str | None = None
    scope: str | None = None
    lookback_days: int | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    history_truncated: bool = False
    machine_id: str | None = None

    @property
    def facts(self) -> tuple[dict[str, Any], ...]:
        """The facts-only projection of the context - the ONLY data a provider reads."""
        return tuple(project_assessment(snapshot) for snapshot in self.context)


@dataclass(frozen=True)
class ProviderResult:
    """A raw generation result: a candidate plus usage metadata.

    ``candidate`` is NOT validated evidence: the route must still run the matching
    grounding validator before the candidate can ever be persisted as ``validated``.
    """

    candidate: Any
    usage: ProviderUsage = field(default_factory=ProviderUsage)
    provider: str = "template"
    model: str | None = None
    raw_text: str | None = None


@runtime_checkable
class AIProvider(Protocol):
    """The provider-neutral generation interface."""

    name: str
    model: str | None
    feature: ProviderFeature

    def build_request(self, context: Any, **kwargs: Any) -> ProviderRequest: ...

    def generate(self, request: ProviderRequest) -> ProviderResult: ...

    def prompt_sha256(self, request: ProviderRequest) -> str: ...


class TemplateAIProvider:
    """Adapter exposing the existing deterministic template providers neutrally.

    It delegates every generation and prompt-digest call to the unchanged template
    provider class for its feature, so the observable behavior (candidate payload,
    prompt digest, validation status, persisted record) is byte-identical.
    """

    def __init__(self, feature: ProviderFeature, impl: Any) -> None:
        self.feature = feature
        self._impl = impl

    # -- identity ----------------------------------------------------------------
    @property
    def name(self) -> str:
        return str(getattr(self._impl, "provider", "template"))

    @property
    def model(self) -> str | None:
        return getattr(self._impl, "model", None)

    # -- request construction ----------------------------------------------------
    def build_request(
        self,
        context: Any,
        *,
        question: str | None = None,
        scope: str | None = None,
        lookback_days: int | None = None,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        history_truncated: bool = False,
        machine_id: str | None = None,
    ) -> ProviderRequest:
        return ProviderRequest(
            feature=self.feature,
            context=tuple(context),
            question=question,
            scope=scope,
            lookback_days=lookback_days,
            window_start=window_start,
            window_end=window_end,
            history_truncated=history_truncated,
            machine_id=machine_id,
        )

    # -- generation --------------------------------------------------------------
    def generate(self, request: ProviderRequest) -> ProviderResult:
        if request.feature is not self.feature:
            raise ProviderError(
                ProviderFailureClass.PROVIDER_ERROR,
                f"request feature '{request.feature.value}' does not match provider "
                f"feature '{self.feature.value}'",
            )
        started = time.perf_counter()
        candidate = self._generate_candidate(request)
        usage = ProviderUsage(latency_ms=(time.perf_counter() - started) * 1000.0)
        return ProviderResult(
            candidate=candidate, usage=usage, provider=self.name, model=self.model
        )

    def _generate_candidate(self, request: ProviderRequest) -> Any:
        if self.feature is ProviderFeature.EXPLAIN:
            return self._impl.explain(request.context[0])
        if self.feature is ProviderFeature.EVIDENCE_REVIEW:
            return self._impl.review(request.context[0])
        if self.feature is ProviderFeature.ANALYZE:
            return self._impl.analyze(request.context[0])
        if self.feature is ProviderFeature.REPORT:
            return self._impl.report(
                list(request.context),
                window_start=request.window_start,
                window_end=request.window_end,
                history_truncated=request.history_truncated,
                lookback_days=request.lookback_days,
            )
        if self.feature is ProviderFeature.ASK:
            return self._impl.ask(
                list(request.context),
                scope=request.scope,
                question=request.question,
                lookback_days=request.lookback_days,
                window_start=request.window_start,
                window_end=request.window_end,
                history_truncated=request.history_truncated,
                machine_id=request.machine_id,
            )
        raise ProviderError(  # pragma: no cover - the enum is exhaustive
            ProviderFailureClass.PROVIDER_ERROR, f"unsupported feature '{self.feature}'"
        )

    # -- prompt digest -----------------------------------------------------------
    def prompt_sha256(self, request: ProviderRequest) -> str:
        if self.feature in (
            ProviderFeature.EXPLAIN,
            ProviderFeature.EVIDENCE_REVIEW,
            ProviderFeature.ANALYZE,
        ):
            return self._impl.prompt_sha256(request.context[0])
        if self.feature is ProviderFeature.REPORT:
            return self._impl.prompt_sha256(
                request.machine_id,
                list(request.context),
                window_start=request.window_start,
                window_end=request.window_end,
                history_truncated=request.history_truncated,
                lookback_days=request.lookback_days,
            )
        if self.feature is ProviderFeature.ASK:
            return self._impl.prompt_sha256(
                list(request.context),
                scope=request.scope,
                question=request.question,
                lookback_days=request.lookback_days,
                machine_id=request.machine_id,
            )
        raise ProviderError(  # pragma: no cover - the enum is exhaustive
            ProviderFailureClass.PROVIDER_ERROR, f"unsupported feature '{self.feature}'"
        )


# Feature -> existing deterministic template provider class. Only the template
# provider exists in this slice; future providers plug in here behind approval.
_TEMPLATE_IMPLS: Mapping[ProviderFeature, type] = {
    ProviderFeature.EXPLAIN: TemplateExplainProvider,
    ProviderFeature.EVIDENCE_REVIEW: TemplateEvidenceReviewProvider,
    ProviderFeature.ANALYZE: TemplateAnalyzeProvider,
    ProviderFeature.REPORT: TemplateReportProvider,
    ProviderFeature.ASK: TemplateAskProvider,
}


def coerce_feature(feature: ProviderFeature | str) -> ProviderFeature:
    """Accept an enum, a kind/"evidence_review", or a path-style "evidence-review"."""
    if isinstance(feature, ProviderFeature):
        return feature
    normalized = str(feature).strip().lower().replace("-", "_")
    try:
        return ProviderFeature(normalized)
    except ValueError as exc:
        raise ValueError(f"unknown AI provider feature: {feature!r}") from exc


def resolve_provider_name(provider: str | None = None, settings: Settings | None = None) -> str:
    """Resolve the configured provider name: explicit override > settings > environment.

    Only a provider NAME is handled here - never a credential or a secret.
    """
    if provider is None:
        provider = settings.ai_provider if settings is not None else get_settings().ai_provider
    return (provider or "").strip().lower()


def get_ai_provider(
    feature: ProviderFeature | str,
    *,
    provider: str | None = None,
    settings: Settings | None = None,
) -> Any:
    """Return the provider for one feature, honouring the configured selection.

    Slice S3: two providers exist. ``template`` is the deterministic offline provider
    for all five features. ``amd`` is the AMD Token Factory static-key provider and is
    approved for the **explain**, **analyze** and **evidence_review** features ONLY -
    the other two features keep the template provider (the caller then reports
    ``provider="template"``, so a template result is never mislabelled as AMD-generated).
    An unknown provider name, or an AMD selection without a usable key/model, raises
    :class:`ProviderError` (fail closed - there is never a silent downgrade for an
    explicitly requested, unavailable provider).
    """
    feat = coerce_feature(feature)
    name = resolve_provider_name(provider, settings)
    if name not in SUPPORTED_AI_PROVIDERS:
        hint = " (planned, not implemented yet)" if name in PLANNED_AI_PROVIDERS else ""
        raise ProviderError(
            ProviderFailureClass.PROVIDER_ERROR,
            f"AI provider '{name}' is not available{hint}; "
            f"supported values: {', '.join(sorted(SUPPORTED_AI_PROVIDERS))}",
        )
    if name == "amd" and feat is ProviderFeature.EXPLAIN:
        # Imported lazily so this routing module stays free of any transport/httpx
        # dependency (the AMD client lives in its own module).
        from .ai_amd_provider import AmdExplainProvider

        return AmdExplainProvider(settings if settings is not None else get_settings())
    if name == "amd" and feat is ProviderFeature.ANALYZE:
        # Same bounded AMD client; the analyze feature has its own instructions and
        # prose schema. Imported lazily for the same reason as explain.
        from .ai_amd_provider import AmdAnalyzeProvider

        return AmdAnalyzeProvider(settings if settings is not None else get_settings())
    if name == "amd" and feat is ProviderFeature.EVIDENCE_REVIEW:
        # Same bounded AMD client; the evidence-review feature has its own instructions
        # and prose schema. Imported lazily for the same reason as explain.
        from .ai_amd_provider import AmdReviewProvider

        return AmdReviewProvider(settings if settings is not None else get_settings())
    impl_cls = _TEMPLATE_IMPLS.get(feat)
    if impl_cls is None:  # pragma: no cover - every feature is mapped
        raise ValueError(f"no provider implementation for feature: {feat.value!r}")
    return TemplateAIProvider(feat, impl_cls())


__all__ = [
    "AIProvider",
    "ProviderCancelled",
    "ProviderError",
    "ProviderFailureClass",
    "ProviderFeature",
    "ProviderRequest",
    "ProviderResult",
    "ProviderTimeout",
    "ProviderUsage",
    "TemplateAIProvider",
    "coerce_feature",
    "get_ai_provider",
    "resolve_provider_name",
]
