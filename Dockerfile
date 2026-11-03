# syntax=docker/dockerfile:1

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
COPY pyproject.toml uv.lock README.md LICENSE ./
# Dependencies first (cache-friendly), project last.
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
    PORT=8000 \
    DIPCATCHER_BIND_PORT=8000
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin dipcatcher
WORKDIR /app
COPY --from=builder --chown=dipcatcher:dipcatcher /app/.venv /app/.venv
COPY --chown=dipcatcher:dipcatcher src ./src
COPY --chown=dipcatcher:dipcatcher configs ./configs
COPY --chown=dipcatcher:dipcatcher docker/healthcheck.py /app/healthcheck.py
COPY --chown=dipcatcher:dipcatcher docker/entrypoint.py /app/entrypoint.py
COPY --chown=dipcatcher:dipcatcher deploy/observability/slo.yml /app/deploy/observability/slo.yml
RUN mkdir -p /app/data && chown dipcatcher:dipcatcher /app/data
USER dipcatcher
ENV DIPCATCHER_DATA_ROOT=/app/data \
    DIPCATCHER_REQUIRE_DATA_WRITE=1
EXPOSE 8000
STOPSIGNAL SIGTERM
# Loopback by default. A published port needs DIPCATCHER_BIND_HOST=0.0.0.0
# and QUANT_API_KEY. The probe runs inside the container network namespace.
# The entrypoint refuses a data directory uid 10001 cannot write.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["/app/.venv/bin/python", "/app/healthcheck.py"]
ENTRYPOINT ["/app/.venv/bin/python", "/app/entrypoint.py"]
CMD ["dipcatcher", "api"]
