# syntax=docker/dockerfile:1.7
# CPU training image (works anywhere). For AMD GPUs use ml/Dockerfile.rocm.
# Build context: repository root. The repo is bind-mounted at /workspace by docker compose,
# so code edits are picked up without rebuilding.
#   docker compose --profile train run --rm trainer telltale-train-sensor

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TELLTALE_DATA_DIR=/workspace/data \
    TELLTALE_MODELS_DIR=/workspace/models
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libsndfile1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace

# CPU-only torch wheel (~200 MB instead of ~2 GB of CUDA libs).
RUN --mount=type=cache,target=/root/.cache/pip pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY ml/pyproject.toml ml/README.md ml/
COPY ml/src ml/src
RUN --mount=type=cache,target=/root/.cache/pip pip install -e "./ml[train,dev]"
COPY AGENTS.md ./
COPY ml/configs ml/configs

CMD ["bash"]
