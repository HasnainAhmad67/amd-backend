"""Runtime settings. Every value can be set with a TELLTALE_* environment variable (see .env.example)."""

import tempfile
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from telltale_ml.paths import data_dir, models_dir

PACKAGE_DATA = Path(__file__).resolve().parent.parent / "data"

# AI provider selection. Two providers exist: the deterministic offline template
# provider (all five features) and the AMD Token Factory static-key provider (slice
# S3, /api/v1/ai/explain ONLY). Any other value fails closed at configuration
# validation - an explicitly selected but unavailable provider is NEVER silently
# downgraded to template. The planned names are listed only to give a clearer error.
SUPPORTED_AI_PROVIDERS = frozenset({"template", "amd"})
PLANNED_AI_PROVIDERS = frozenset({"openai"})

# AMD Token Factory shared endpoint (documented base URL). Providers are restricted to
# this HTTPS host and fixed base path; the transport appends only /chat/completions.
AMD_API_BASE_URL_DEFAULT = "https://developer.amd.com.cn/radeon/api/v1"
AMD_HOST = "developer.amd.com.cn"
AMD_BASE_PATH = "/radeon/api/v1"

# Bounded transport limits (fail closed above the maxima).
AI_TIMEOUT_MAX_S = 120.0
AI_MAX_OUTPUT_TOKENS_MAX = 4000
AI_MAX_RETRIES_MAX = 3
AI_RETRY_MAX_S_MAX = 30.0


def is_allowed_amd_base_url(value: str) -> bool:
    """True only for https://developer.amd.com.cn/radeon/api/v1 (no query/fragment).

    Keeps the AMD request pinned to the documented HTTPS host and fixed base path, so
    a mis-set environment variable can never redirect the bearer key elsewhere.
    """
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.hostname == AMD_HOST
        and parsed.port is None
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and parsed.path.rstrip("/") == AMD_BASE_PATH
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TELLTALE_", extra="ignore")

    env: str = "dev"
    models_dir: Path = Field(default_factory=models_dir)
    data_dir: Path = Field(default_factory=data_dir)
    # JSON list in env, e.g. TELLTALE_CORS_ORIGINS='["http://localhost:5173"]'
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8080"]

    # Demo fleet simulator (replays AI4I-style physics so the dashboard is alive without hardware).
    simulator_enabled: bool = True
    simulator_interval_s: float = 3.0
    simulator_seed: int = 7
    fleet_file: Path = PACKAGE_DATA / "fleet.yaml"
    playbook_file: Path = PACKAGE_DATA / "playbook.yaml"

    # First start: if there is no sensor model yet, train one automatically in the background
    # (downloads the 0.5 MB AI4I dataset; falls back to simulated data when offline).
    sensor_autotrain: bool = True
    sensor_autotrain_offline: bool = False  # skip the download and train on simulated data
    fallback_models_dir: Path = Path(tempfile.gettempdir()) / "telltale-models"

    max_upload_mb: float = 25.0

    # Keep alerts across restarts in a small SQLite file (default: <data_dir>/state/telltale.db).
    persist_alerts: bool = True
    state_db: Path | None = None

    # Demo authentication for the NEW /api/v1/assessments routes ONLY (legacy
    # routes stay outside this boundary). Strict JSON object {"<actor>": "<token>"}.
    # Empty/unset => the assessments routes deny every request (default-deny).
    # Local demo only - NOT production authentication. Tokens are never logged
    # or echoed; X-Actor-ID headers are ignored.
    api_tokens: str = ""

    # When TELLTALE_DEMO_MODE=1 and no explicit api_tokens are set,
    # use a fixed demo token and owner. The normal auth behavior is preserved
    # when demo_mode is False or explicit tokens are configured.

    # Optional LLM (any OpenAI-compatible server, e.g. vLLM on an AMD Instinct GPU).
    # Unset -> deterministic template summaries. The LLM only rephrases model outputs; it never scores risk.
    llm_base_url: str | None = None
    llm_api_key: str = "not-needed"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_timeout_s: float = 20.0

    # Which generator the NEW Central AI Agent features use. "template" (default) is the
    # deterministic offline provider for all five features. "amd" selects the AMD Token
    # Factory static-key provider for /api/v1/ai/explain ONLY - the other four features
    # keep the template provider. The value is a provider NAME, not a secret, so echoing
    # it is safe.
    ai_provider: str = "template"

    # Demo mode for local development without login.
    # When TELLTALE_DEMO_MODE=1, all assessment/AI output routes use a fixed
    # local owner "local-demo-user" and a fixed demo token, bypassing the
    # normal TELLTALE_API_TOKENS authentication requirement. When disabled,
    # the normal auth is enforced strictly.
    demo_mode: bool = False

    # Demo mode for local development without login.
    # When TELLTALE_DEMO_MODE=1, all assessment/AI output routes use a fixed
    # local owner "local-demo-user" and a fixed demo token, bypassing the
    # normal TELLTALE_API_TOKENS authentication requirement. When disabled,
    # the normal auth is enforced strictly.
    # AMD Token Factory (static Bearer API key; slice S3). The key is a SecretStr: it is
    # never logged, echoed, serialized into evidence or persisted. Required (together
    # with a model) only when ai_provider == "amd".
    amd_api_base_url: str = AMD_API_BASE_URL_DEFAULT
    amd_api_key: SecretStr | None = None
    amd_model: str | None = None

    # Bounded AMD transport knobs. Values above the documented maxima fail closed.
    ai_timeout_s: float = 30.0
    # OPTIONAL max output tokens for the AMD chat-completions request. Unset (or "", "auto",
    # "omit", "none") OMITS ``max_tokens`` from the body: some AMD models reject an explicit
    # max_tokens with HTTP 400 (DeepSeek-V4-Flash reports its allowed maximum as null). A
    # positive integer sends that value; anything outside 1..MAX fails closed.
    ai_max_output_tokens: int | None = None
    ai_max_retries: int = 1
    ai_retry_max_s: float = 2.0

    @field_validator("ai_provider")
    @classmethod
    def _validate_ai_provider(cls, value: str) -> str:
        name = (value or "").strip().lower()
        if name not in SUPPORTED_AI_PROVIDERS:
            hint = " (planned, not implemented yet)" if name in PLANNED_AI_PROVIDERS else ""
            supported = ", ".join(sorted(SUPPORTED_AI_PROVIDERS))
            raise ValueError(
                f"TELLTALE_AI_PROVIDER '{name}' is not supported{hint}; supported values: {supported}"
            )
        return name

    @field_validator("amd_api_base_url")
    @classmethod
    def _validate_amd_base_url(cls, value: str) -> str:
        # The error never echoes the value (a mistyped URL could contain a secret).
        if not is_allowed_amd_base_url(value):
            raise ValueError(
                f"TELLTALE_AMD_API_BASE_URL must be https://{AMD_HOST}{AMD_BASE_PATH} "
                "(https, no query/fragment, no credentials)"
            )
        return value

    @field_validator("ai_timeout_s")
    @classmethod
    def _validate_ai_timeout_s(cls, value: float) -> float:
        if not (0 < float(value) <= AI_TIMEOUT_MAX_S):
            raise ValueError(f"TELLTALE_AI_TIMEOUT_S must be > 0 and <= {AI_TIMEOUT_MAX_S:g}")
        return float(value)

    @field_validator("ai_max_output_tokens", mode="before")
    @classmethod
    def _validate_ai_max_output_tokens(cls, value: object) -> int | None:
        """None/""/auto/omit/none => omit max_tokens; a positive int => send it; else fail closed."""
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(
                "TELLTALE_AI_MAX_OUTPUT_TOKENS must be a positive integer, or unset/auto/omit"
            )
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("", "auto", "omit", "none"):
                return None
            try:
                value = int(text)
            except ValueError:
                raise ValueError(
                    "TELLTALE_AI_MAX_OUTPUT_TOKENS must be a positive integer, or unset/auto/omit"
                ) from None
        if isinstance(value, float) and not float(value).is_integer():
            raise ValueError(
                "TELLTALE_AI_MAX_OUTPUT_TOKENS must be a positive integer, or unset/auto/omit"
            )
        try:
            number = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise ValueError(
                "TELLTALE_AI_MAX_OUTPUT_TOKENS must be a positive integer, or unset/auto/omit"
            ) from None
        if not (1 <= number <= AI_MAX_OUTPUT_TOKENS_MAX):
            raise ValueError(
                f"TELLTALE_AI_MAX_OUTPUT_TOKENS must be between 1 and {AI_MAX_OUTPUT_TOKENS_MAX}"
            )
        return number

    @field_validator("ai_max_retries")
    @classmethod
    def _validate_ai_max_retries(cls, value: int) -> int:
        if not (0 <= int(value) <= AI_MAX_RETRIES_MAX):
            raise ValueError(f"TELLTALE_AI_MAX_RETRIES must be between 0 and {AI_MAX_RETRIES_MAX}")
        return int(value)

    @field_validator("ai_retry_max_s")
    @classmethod
    def _validate_ai_retry_max_s(cls, value: float) -> float:
        if not (0 <= float(value) <= AI_RETRY_MAX_S_MAX):
            raise ValueError(
                f"TELLTALE_AI_RETRY_MAX_S must be between 0 and {AI_RETRY_MAX_S_MAX:g}"
            )
        return float(value)

    @model_validator(mode="after")
    def _require_amd_credentials(self) -> "Settings":
        # Fail closed at startup: selecting AMD without a key and a model is a
        # configuration error. The message never contains the key value.
        if self.ai_provider == "amd":
            if self.amd_api_key is None or not self.amd_api_key.get_secret_value().strip():
                raise ValueError("TELLTALE_AMD_API_KEY is required when TELLTALE_AI_PROVIDER=amd")
            if not (self.amd_model or "").strip():
                raise ValueError("TELLTALE_AMD_MODEL is required when TELLTALE_AI_PROVIDER=amd")
        return self

    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "samples"


@lru_cache
def get_settings() -> Settings:
    return Settings()
