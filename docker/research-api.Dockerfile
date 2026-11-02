# syntax=docker/dockerfile:1

# Read-only research evidence API (quant_fund.api.research_api).
# Serves committed receipts/, verifier/, artifacts/ plus the gitignored
# data/metadata tree — mount a data root at runtime:
#
#   docker build -f docker/research-api.Dockerfile -t dipcatcher-research-api .
#   docker run --rm -p 127.0.0.1:8010:8010 \
#     -e RESEARCH_API_KEY \
#     -v /path/to/data:/app/data:ro dipcatcher-research-api
#
# Export RESEARCH_API_KEY before launching; the container requires it.
# The published host port is loopback-only in the example. The service has no
# trading, order, broker, or execution endpoints.
#
# Context: receipts/, verifier/ and artifacts/ (COPY below) stay in the shared
# build context on purpose — the root .dockerignore deliberately does not
# exclude them, so this build works with any builder and any daemon version
# (Lead decision 2026-10-08: one fail-safe ignore file, no per-Dockerfile
# ignore-file convention).

# --- Builder: resolve/sync the locked venv, then install the project ---
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# uv 0.11.23 wheels and sdist, hash-pinned in docker/uv-requirements.txt.
COPY docker/uv-requirements.txt /tmp/uv-requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes --no-deps \
        -r /tmp/uv-requirements.txt \
    && rm /tmp/uv-requirements.txt
# LICENSE is mandatory here: pyproject has `license = { file = "LICENSE" }` and
# hatchling aborts the in-image wheel build (`uv sync --no-editable`) with
# "OSError: License file does not exist: LICENSE" when it is absent — proven
# locally before this line was fixed (2026-10-08, lane L5).
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# --- Runtime: no uv, no pip, no build caches; self-contained venv ---
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PORT=8010 \
    RESEARCH_API_HOST=0.0.0.0 \
    RESEARCH_API_PORT=8010 \
    RESEARCH_API_DATA_ROOT=/app/data \
    RESEARCH_API_RECEIPTS_DIR=/app/receipts \
    RESEARCH_API_VERIFIER_DIR=/app/verifier \
    RESEARCH_API_ARTIFACTS_DIR=/app/artifacts
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin dipcatcher
WORKDIR /app
COPY --from=builder --chown=dipcatcher:dipcatcher /app/.venv /app/.venv
COPY --chown=dipcatcher:dipcatcher src ./src
COPY --chown=dipcatcher:dipcatcher receipts ./receipts
COPY --chown=dipcatcher:dipcatcher verifier ./verifier
COPY --chown=dipcatcher:dipcatcher artifacts ./artifacts
# data/ is dockerignored — mount it at runtime (see header).
COPY --chown=dipcatcher:dipcatcher docker/healthcheck.py /app/healthcheck.py
COPY --chown=dipcatcher:dipcatcher deploy/observability/slo.yml /app/deploy/observability/slo.yml
RUN mkdir -p /app/data && chown dipcatcher:dipcatcher /app/data
USER dipcatcher
EXPOSE 8010
STOPSIGNAL SIGTERM
# RESEARCH_API_HOST is 0.0.0.0, so the process refuses to start without
# RESEARCH_API_KEY. The probe runs inside the container network namespace.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["/app/.venv/bin/python", "/app/healthcheck.py"]
CMD ["python", "-m", "quant_fund.api.research_api"]
