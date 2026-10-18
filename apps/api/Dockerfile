# syntax=docker/dockerfile:1.7
# Build context: repository root (the API depends on the local ml/ package).
#   docker build -f apps/api/Dockerfile --target prod -t telltale-api .

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TELLTALE_MODELS_DIR=/models \
    TELLTALE_DATA_DIR=/data \
    TELLTALE_CONFIG_DIR=/app/ml/configs
# libgomp1: LightGBM (OpenMP) · libsndfile1: audio decoding · curl: healthcheck
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libsndfile1 curl \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# ---- dev: editable install, source is bind-mounted by docker-compose.dev.yml ----
FROM base AS dev
COPY ml/pyproject.toml ml/README.md ml/
COPY ml/src ml/src
COPY ml/configs ml/configs
COPY apps/api/pyproject.toml apps/api/
COPY apps/api/telltale_api apps/api/telltale_api
RUN --mount=type=cache,target=/root/.cache/pip pip install -e ./ml -e "./apps/api[dev]"
EXPOSE 8000
CMD ["uvicorn", "telltale_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/apps/api/telltale_api", "--reload-dir", "/app/ml/src"]

# ---- builder: install into /install, then slim it down ----
FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends binutils && rm -rf /var/lib/apt/lists/*
COPY ml/pyproject.toml ml/README.md ml/
COPY ml/src ml/src
COPY apps/api/pyproject.toml apps/api/
COPY apps/api/telltale_api apps/api/telltale_api
RUN --mount=type=cache,target=/root/.cache/pip pip install --prefix=/install ./ml ./apps/api
# Packages ship their own test suites (~180 MB) and unstripped native libraries; neither is needed at runtime.
RUN SITE=/install/lib/python3.11/site-packages \
 && find "$SITE" -type d \( -name tests -o -name test \) -prune -exec rm -rf {} + \
 && find "$SITE" -type d -name __pycache__ -prune -exec rm -rf {} + \
 && find "$SITE" -name "*.so*" -type f -exec strip --strip-unneeded {} + 2>/dev/null; true

# ---- prod: only the installed packages + configs ----
FROM base AS prod
COPY --from=builder /install /usr/local
COPY ml/configs ml/configs
# Runs as root on purpose: ./models is shared with the trainer container (also root) and the API writes into it
# (first-start sensor model, sound enrollment). A non-root user here hits bind-mount permission errors on Linux
# and Docker Desktop. This is a local demo app; if you deploy it publicly, add a dedicated user and fix ownership.
RUN mkdir -p /models /data
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
  CMD curl -fsS http://localhost:8000/api/v1/health || exit 1
CMD ["uvicorn", "telltale_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
