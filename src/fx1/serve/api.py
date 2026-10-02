"""The fx-1 harness HTTP surface.

``fx1.harness`` registers every lab command the model may touch; this app
exposes that registry — plus receipt verification and backend completion —
over HTTP so an fx-1 instance drives the harness through one authenticated
socket instead of shell access.

Routes (all POST bodies are ``extra="forbid"`` — no silent arguments):

- ``GET  /health`` — liveness + which backends are configured (booleans
  only; credential values never leave this process).
- ``GET  /harness/commands`` — the registry; ``?role=`` filters by
  HarnessRole. Anything not listed is unreachable — fail-closed by
  construction, same as the in-process surface.
- ``POST /harness/runs`` — execute a registered command through
  :class:`~fx1.harness.Harness`; unknown names 404, ``configs/`` escapes
  422, and the command's own timeout bounds the call.
- ``POST /harness/complete`` — chat completion through
  ``get_backend(backend)`` wrapped by ``cited_complete`` (the honesty gate
  runs server-side, in production, on every response). Backend config
  failures 503, unknown backends 404, honesty violations 502 — the model's
  output contract is enforced before bytes leave.
- ``POST /harness/complete/batch`` — many gated completions over one shared
  backend instance; per-item ok/error verdicts, never a 5xx per item.
- ``POST /receipts/verify`` — verify an arbitrary receipt object with
  ``verify_receipt_payload``; callers never need filesystem access to the
  evidence store.

Auth posture mirrors ``quant_fund.api.research_api``: ``/health`` is the
only unauthenticated route; when ``FX1_API_KEY`` is set every other route
requires ``X-API-Key``; unset, only loopback clients are served.
"""

from __future__ import annotations

import builtins
import hmac
import json
import logging
import math
import os
import queue
import re
import threading
import time
import urllib.parse
import urllib.request
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import (
    AbstractAsyncContextManager,
    asynccontextmanager,
    contextmanager,
)
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from starlette.middleware.gzip import GZipMiddleware

from fx1 import __version__
from fx1.harness import Harness, HarnessRole
from fx1.honesty import Fx1HonestyError, validate_fx1_output
from fx1.serve.backends import (
    BYOK_API_KEY_ENV,
    BYOK_BASE_URL_ENV,
    BYOK_MODEL_ENV,
    LOCAL_SERVE_CMD_ENV,
    LOCAL_SERVE_URL_ENV,
    BackendNotConfiguredError,
    StreamingBackend,
    get_backend,
)
from fx1.serve.chat import cited_complete
from fx1.serve.contract import API_VERSION
from fx1.serve.webhooks import (
    WEBHOOK_SIGNATURE_HEADER,
    WEBHOOK_TIMESTAMP_HEADER,
    sign_webhook,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload

_API_KEY_ENV = "FX1_API_KEY"
_MAX_INFLIGHT_ENV = "FX1_API_MAX_INFLIGHT"
_SSE_KEEPALIVE_ENV = "FX1_API_SSE_KEEPALIVE_S"
_IDEM_MAX_ENV = "FX1_API_IDEM_MAX"
_IDEM_KEY_MAX = 256
# One POST verifies a whole receipt bundle — fx-1's gating path handles
# dozens per task; still bounded so a hostile bundle can't pin a worker.
_VERIFY_BATCH_MAX = 64
_JOB_MAX_ENV = "FX1_API_JOB_MAX"
_RATE_LIMIT_ENV = "FX1_API_RATE_LIMIT_RPS"
_GZIP_MIN_ENV = "FX1_API_GZIP_MIN_BYTES"
_CORS_ORIGINS_ENV = "FX1_API_CORS_ORIGINS"
_BREAKER_THRESHOLD_ENV = "FX1_API_BREAKER_THRESHOLD"
_BREAKER_COOLDOWN_ENV = "FX1_API_BREAKER_COOLDOWN_S"

# Headers browser clients can read off responses when CORS is enabled.
_CORS_EXPOSE_HEADERS = [
    "Location",
    "Retry-After",
    "X-Fx1-Api-Version",
    "X-RateLimit-Limit",
    "X-RateLimit-Remaining",
    "X-RateLimit-Reset",
    "X-Request-ID",
]
_CORS_ALLOW_HEADERS = [
    "Content-Type",
    "Idempotency-Key",
    "X-API-Key",
    "X-Request-ID",
]
_RATE_LIMIT_KEYS_MAX = 4096


def _version_info() -> VersionResponse:
    """Version negotiation: the wire contract + package release."""
    return VersionResponse(api_version=API_VERSION, fx1_version=__version__)


_PUBLIC_PATHS = frozenset({"/health"})

# Response headers the middleware stamps on every response — declared on the
# OpenAPI spec so generated clients see them typed instead of having to know.
_DECLARED_COMMON_HEADERS: dict[str, dict[str, Any]] = {
    "X-Request-ID": {
        "schema": {"type": "string"},
        "description": "Request id — echoed from the inbound X-Request-ID or minted.",
    },
    "X-Fx1-Api-Version": {
        "schema": {"type": "string"},
        "description": "Wire-contract version; clients gate on it via /harness/version.",
    },
    "X-Content-Type-Options": {
        "schema": {"type": "string"},
        "description": "Always `nosniff`.",
    },
    "Cache-Control": {"schema": {"type": "string"}, "description": "Always `no-store`."},
    "Referrer-Policy": {
        "schema": {"type": "string"},
        "description": "Always `no-referrer`.",
    },
}
_DECLARED_RATELIMIT_HEADERS: dict[str, dict[str, Any]] = {
    "X-RateLimit-Limit": {
        "schema": {"type": "integer"},
        "description": "Configured request budget per second (present only when the "
        "rate limiter is enabled).",
    },
    "X-RateLimit-Remaining": {
        "schema": {"type": "integer"},
        "description": "Tokens left in this client's bucket after this response.",
    },
    "X-RateLimit-Reset": {
        "schema": {"type": "integer"},
        "description": "Seconds until the bucket refills.",
    },
}
_DECLARED_RETRY_AFTER: dict[str, Any] = {
    "schema": {"type": "integer"},
    "description": "Seconds to wait before retrying (429 rate-limit and 503 capacity responses).",
}
_DECLARED_LOCATION: dict[str, Any] = {
    "schema": {"type": "string"},
    "description": "URL of the created job's status endpoint.",
}


def _declare_response_headers(app: FastAPI, rate_limited: bool) -> None:
    """Materialize the cached spec once and stamp the headers the middleware
    actually sets — generated clients inherit the contract instead of guessing."""
    spec = app.openapi()
    for item in spec.get("paths", {}).values():
        for op in item.values():
            if not isinstance(op, dict):
                continue
            for code, resp in op.get("responses", {}).items():
                if not isinstance(resp, dict):
                    continue
                hdrs = resp.setdefault("headers", {})
                hdrs.update(_DECLARED_COMMON_HEADERS)
                if rate_limited:
                    hdrs.update(_DECLARED_RATELIMIT_HEADERS)
                if code in ("429", "503"):
                    hdrs.setdefault("Retry-After", _DECLARED_RETRY_AFTER)
            if op.get("operationId") == "submit_job":
                op["responses"]["202"].setdefault("headers", {})["Location"] = _DECLARED_LOCATION


# Canonical machine code for unambiguous statuses; ambiguous statuses
# (three different 503s, two 502s) carry an explicit ApiError code.
_STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "too_large",
    422: "validation",
    429: "too_many_requests",
    500: "internal",
    501: "not_implemented",
    503: "unavailable",
}


class ApiError(HTTPException):
    """HTTPException carrying a stable machine ``code``, surfaced in the
    ``{"detail", "code"}`` error envelope so clients switch on it instead
    of matching detail text."""

    def __init__(
        self,
        status_code: int,
        detail: Any,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code, detail, headers=headers)
        self.code = code or _STATUS_CODES.get(status_code, "internal")


def _err_code(exc: HTTPException) -> str:
    if isinstance(exc, ApiError):
        return exc.code
    return _STATUS_CODES.get(exc.status_code, "internal")


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})
_MAX_BODY_BYTES = 1 << 20
# Async job results persist in the store until eviction — stdout/stderr
# are capped per field so one chatty command can't pin unbounded memory.
_JOB_RESULT_MAX_BYTES = 1 << 20


def _cap_job_text(text: str) -> tuple[str, bool]:
    """Bound a stored job field; truncate at the byte cap, flag honestly."""
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= _JOB_RESULT_MAX_BYTES:
        return text, False
    return raw[:_JOB_RESULT_MAX_BYTES].decode("utf-8", errors="ignore"), True


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(_Model):
    status: Literal["ok"] = "ok"
    service: str = "fx1-harness-api"
    version: str = __version__
    registered_commands: int
    backends: dict[str, bool]
    draining: bool = False


class HarnessCommandItem(_Model):
    name: str
    role: str
    argv: list[str]
    timeout_s: int
    description: str


class HarnessCommandListResponse(_Model):
    items: list[HarnessCommandItem]
    total: int


class HarnessRunRequest(_Model):
    command: str
    extra_args: list[str] = Field(default_factory=list, max_length=64)
    config: str | None = None
    callback_url: str | None = None
    # HMAC signing secret for the callback delivery — never echoed on the
    # job record (stored as a PrivateAttr, excluded from serialization).
    callback_secret: str | None = None
    # In-body dedup key: on the single-submit route the Idempotency-Key
    # header wins when both are present; on batch submit this is the only
    # channel. Excluded from the idempotency fingerprint (it's transport,
    # not payload semantics).
    idempotency_key: str | None = Field(default=None, max_length=_IDEM_KEY_MAX)

    @field_validator("callback_url")
    @classmethod
    def _callback_url_http(cls, v: str | None) -> str | None:
        """Webhook target must be a real http(s) URL — the job record is
        POSTed to it on every terminal transition."""
        if v is None:
            return v
        parsed = urllib.parse.urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"callback_url must be an http(s) URL with a host, got {v!r}")
        return v

    @model_validator(mode="after")
    def _callback_secret_needs_url(self) -> HarnessRunRequest:
        if self.callback_secret is not None and not self.callback_url:
            raise ValueError("callback_secret requires callback_url")
        return self


class HarnessRunResponse(_Model):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    ok: bool
    timeout_s: int
    replayed: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class ChatMessage(_Model):
    role: str
    content: str


class CompleteRequest(_Model):
    backend: Literal["hosted_k3", "local_fx1", "byok"]
    messages: list[ChatMessage] = Field(min_length=1, max_length=512)
    checkpoint_dir: str | None = None
    receipt_hashes: list[str] | None = None


class CompleteResponse(_Model):
    backend: str
    model: str | None
    content: str
    receipt_hashes: list[str]
    # True when the response came from the Idempotency-Key cache — lets
    # fx-1 audit retried calls without paying for them twice.
    replayed: bool = False


class CompleteBatchRequest(_Model):
    backend: Literal["hosted_k3", "local_fx1", "byok"]
    batch: list[list[ChatMessage]] = Field(min_length=1, max_length=64)
    checkpoint_dir: str | None = None
    receipt_hashes: list[str] | None = None
    max_workers: int = Field(default=4, ge=1, le=16)


class CompleteBatchItem(_Model):
    ok: bool
    content: str | None = None
    error: str | None = None
    error_class: str | None = None


class CompleteBatchResponse(_Model):
    backend: str
    model: str | None
    receipt_hashes: list[str]
    results: list[CompleteBatchItem]
    replayed: bool = False


class ReceiptVerifyRequest(_Model):
    receipt: dict[str, Any]


class ReceiptVerifyResponse(_Model):
    valid: bool
    path: str
    schema_tag: str
    kind: str | None
    verdict: str | None
    digest_convention: str | None
    errors: list[str]
    warnings: list[str]


class ReceiptVerifyBatchRequest(_Model):
    receipts: list[dict[str, Any]] = Field(min_length=1, max_length=_VERIFY_BATCH_MAX)


class ReceiptVerifyBatchItem(ReceiptVerifyResponse):
    """One per-item verdict: the same payload the single-verify route
    returns, plus the index so callers can map results back."""

    index: int


class ReceiptVerifyBatchResponse(_Model):
    verified: int
    failed: int
    results: list[ReceiptVerifyBatchItem]


class MetricsResponse(_Model):
    """Point-in-time ops snapshot: totals since process start."""

    uptime_s: float
    requests_total: int
    errors_total: int
    by_status: dict[str, int]
    inflight: int
    inflight_watermark: int
    max_inflight: int
    draining: bool
    rate_limited_total: int = 0


class DrainResponse(_Model):
    """Result of latching drain mode: state + live in-flight count."""

    draining: bool
    inflight: int
    drained: bool = False


class ReadyResponse(_Model):
    """Readiness probe payload — only emitted while accepting work."""

    ready: Literal[True] = True
    inflight: int


class VersionResponse(_Model):
    """Wire-contract + package versions — the client's negotiation payload."""

    api_version: str
    fx1_version: str


class CapabilitiesResponse(_Model):
    """Self-describing discovery payload: which wire features this build
    serves and the operational limits in effect — clients self-configure
    (batch sizes, retry budgets, stream use) from one call instead of
    hardcoding server internals."""

    api_version: str
    fx1_version: str
    features: dict[str, bool]
    limits: dict[str, float]
    backends: dict[str, bool]
    roles: list[str]


class BackendStatusEntry(_Model):
    """One backend's liveness surface: whether it is configured and, when
    the circuit breaker is enabled, whether it is currently fast-failing."""

    configured: bool
    circuit_open: bool
    cooldown_remaining_s: float
    consecutive_failures: int


_JOB_STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")
_TERMINAL_JOB_STATUS = frozenset({"succeeded", "failed", "cancelled"})
_JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class JobSubmitResponse(_Model):
    job_id: str
    status: _JobStatus
    replayed: bool = False


class JobStatusResponse(_Model):
    """Live job record; ``result`` appears only once status is terminal."""

    job_id: str
    status: _JobStatus
    created_at: float
    finished_at: float | None
    result: HarnessRunResponse | None
    error: str | None
    callback_url: str | None = None
    callback_status: Literal["delivered", "failed"] | None = None
    _callback_secret: str | None = PrivateAttr(default=None)
    callback_error: str | None = None
    callback_attempts: int = 0


class _Metrics:
    """Request counters + inflight gauge, shared via app.state."""

    def __init__(self, max_inflight: int) -> None:
        self.started = time.monotonic()
        self.max_inflight = max_inflight
        self._cond = threading.Condition()
        self._requests_total = 0
        self._errors_total = 0
        self._by_status: dict[int, int] = {}
        self._inflight = 0
        self._watermark = 0
        self._rate_limited_total = 0
        self.draining = threading.Event()

    def record_rate_limited(self) -> None:
        with self._cond:
            self._rate_limited_total += 1

    def record(self, status: int) -> None:
        with self._cond:
            self._requests_total += 1
            self._by_status[status] = self._by_status.get(status, 0) + 1
            if status >= 400:
                self._errors_total += 1

    def acquire(self) -> None:
        with self._cond:
            self._inflight += 1
            self._watermark = max(self._watermark, self._inflight)

    def release(self) -> None:
        with self._cond:
            self._inflight -= 1
            self._cond.notify_all()

    def wait_idle(self, timeout_s: float) -> bool:
        """Block until inflight reaches zero or the timeout lapses — the
        drain-lifecycle wait channel. True when the pool actually emptied."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._inflight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    def snapshot(self) -> MetricsResponse:
        with self._cond:
            return MetricsResponse(
                uptime_s=round(time.monotonic() - self.started, 3),
                requests_total=self._requests_total,
                errors_total=self._errors_total,
                by_status={str(k): v for k, v in sorted(self._by_status.items())},
                inflight=self._inflight,
                inflight_watermark=self._watermark,
                max_inflight=self.max_inflight,
                draining=self.draining.is_set(),
                rate_limited_total=self._rate_limited_total,
            )


def _env_int_bound(name: str, default: int, given: int | None) -> int:
    """Positive-int tunable from arg or env, fail-closed below 1."""
    v = int(os.environ.get(name, str(default))) if given is None else given
    if v < 1:
        raise ValueError(f"{name} bound must be >= 1, got {v}")
    return v


def _env_int_floor(name: str, default: int, given: int | None) -> int:
    """Non-negative-int tunable from arg or env; 0 disables the feature."""
    v = int(os.environ.get(name, str(default))) if given is None else given
    if v < 0:
        raise ValueError(f"{name} bound must be >= 0, got {v}")
    return v


def _env_float_floor(name: str, default: float, given: float | None) -> float:
    """Non-negative float tunable from arg or env, fail-closed below 0."""
    v = float(os.environ.get(name, str(default))) if given is None else given
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got {v}")
    return v


def _idem_lookup[IdemT: BaseModel](
    idempotency_key: str | None,
    store: _IdemStore[IdemT],
    body_fp: str,
) -> tuple[str | None, IdemT | None]:
    """Shared Idempotency-Key preamble: normalize + bound the key, then
    look up a stored replay. Returns ``(key, cached)`` — a cached hit
    is the response to return verbatim plus ``replayed: True``; a key
    reused with a different body fails closed 409."""
    key = (idempotency_key or "").strip() or None
    if key is None:
        return None, None
    if len(key) > _IDEM_KEY_MAX:
        raise ApiError(400, "Idempotency-Key must be <= 256 chars")
    entry = store.get(key)
    if entry is None:
        return key, None
    fp, cached = entry
    if fp != body_fp:
        raise ApiError(409, "Idempotency-Key reuse with a different request body")
    return key, cached.model_copy(update={"replayed": True})


_WEBHOOK_MAX_ATTEMPTS = 3
_WEBHOOK_BACKOFF_S = 0.5


def _deliver_job_callback(job: JobStatusResponse) -> None:
    """Terminal-state webhook: POST the full job record to the caller's
    ``callback_url``. Best-effort — a dead or slow endpoint records
    ``callback_status='failed'`` on the job, never raises into the worker
    and never changes the job's own status. Transient faults (network
    errors, 5xx) retry ``_WEBHOOK_MAX_ATTEMPTS`` times with capped backoff;
    a 4xx is a definitive rejection and is never retried."""
    url = job.callback_url
    if not url:
        return
    for attempt in range(_WEBHOOK_MAX_ATTEMPTS):
        if attempt:
            time.sleep(_WEBHOOK_BACKOFF_S * (1 << (attempt - 1)))
        job.callback_attempts = attempt + 1
        try:
            payload = job.model_dump_json().encode()
            headers = {"Content-Type": "application/json"}
            if job._callback_secret:
                ts = str(int(time.time()))
                headers[WEBHOOK_TIMESTAMP_HEADER] = ts
                headers[WEBHOOK_SIGNATURE_HEADER] = sign_webhook(job._callback_secret, ts, payload)
            req = urllib.request.Request(
                url,
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310  # nosec B310 — caller-declared webhook target, validated http(s) at submit
                if resp.status < 400:
                    job.callback_status = "delivered"
                    job.callback_error = None
                    return
                job.callback_status = "failed"
                job.callback_error = f"callback endpoint returned {resp.status}"
                if 400 <= resp.status < 500:
                    return  # definitive rejection — never retried
        except Exception as exc:  # noqa: BLE001 — delivery faults land on the record, not the worker
            job.callback_status = "failed"
            job.callback_error = f"{type(exc).__name__}: {exc}"


def _submit_job(
    body: HarnessRunRequest,
    idempotency_key: str | None,
    lab: Harness,
    job_store: _JobStore,
    metrics: _Metrics,
    inflight: threading.BoundedSemaphore,
    jobs_executor: ThreadPoolExecutor,
) -> JobSubmitResponse:
    """Job submission core: idempotency lookup -> command validation ->
    slot admission -> background execution. The slot is held for the
    job's lifetime and released by the worker, so the queue can never
    grow past ``max_inflight`` (no unbounded buffering)."""
    key = (idempotency_key or "").strip() or None
    if key is not None and len(key) > _IDEM_KEY_MAX:
        raise ApiError(400, "Idempotency-Key must be <= 256 chars")
    body_fp = body.model_dump_json(exclude={"idempotency_key"})
    if key is not None:
        entry = job_store.get_key(key)
        if entry is not None:
            fp, job_id = entry
            if fp != body_fp:
                raise ApiError(
                    409,
                    "Idempotency-Key reuse with a different request body",
                )
            job = job_store.get(job_id)
            if job is not None:
                return JobSubmitResponse(job_id=job_id, status=job.status, replayed=True)
    try:
        lab.get(body.command)  # fail closed at submit, not in the worker
    except KeyError as exc:
        raise ApiError(404, str(exc)) from exc
    if metrics.draining.is_set():
        raise ApiError(503, "harness is draining — no new work accepted", code="draining")
    if not inflight.acquire(blocking=False):
        raise ApiError(
            503,
            f"harness at max_inflight={metrics.max_inflight} — retry later",
            code="over_capacity",
            headers={"Retry-After": "1"},
        )
    metrics.acquire()
    job = JobStatusResponse(
        job_id=uuid.uuid4().hex,
        status="queued",
        created_at=time.time(),
        finished_at=None,
        result=None,
        error=None,
        callback_url=body.callback_url,
    )
    job._callback_secret = body.callback_secret

    def _exec() -> None:
        if job.status == "cancelled":
            metrics.release()
            inflight.release()
            return
        job.status = "running"
        try:
            result = lab.run(
                body.command,
                body.extra_args or None,
                config=Path(body.config) if body.config else None,
            )
            command = lab.get(body.command)
            stdout, stdout_truncated = _cap_job_text(result.stdout)
            stderr, stderr_truncated = _cap_job_text(result.stderr)
            job.result = HarnessRunResponse(
                command=result.command,
                exit_code=result.exit_code,
                stdout=stdout,
                stderr=stderr,
                ok=result.ok,
                timeout_s=command.timeout_s,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
            )
            job.status = "succeeded"
        except Exception as exc:  # noqa: BLE001 — worker faults land in the record
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = "failed"
        finally:
            _deliver_job_callback(job)
        job.finished_at = time.time()
        metrics.release()
        inflight.release()

    try:
        jobs_executor.submit(_exec)
    except RuntimeError as exc:  # executor gone (shutdown race)
        metrics.release()
        inflight.release()
        raise ApiError(503, "job executor unavailable", code="over_capacity") from exc
    job_store.put(job, key, body_fp)
    return JobSubmitResponse(job_id=job.job_id, status=job.status, replayed=False)


def _make_lifespan(
    metrics: _Metrics,
    job_store: _JobStore,
    jobs_executor: ThreadPoolExecutor,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Graceful-exit contract: on shutdown the gate drains (new work gets
    503), every still-queued job flips to 'cancelled' and fires its
    signed webhook, and the executor releases pending futures. Running
    jobs aren't interrupted — they finish bounded by their command
    timeout or die with the process."""

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        metrics.draining.set()
        for pending in job_store.cancel_pending():
            _deliver_job_callback(pending)
        jobs_executor.shutdown(wait=False, cancel_futures=True)

    return _lifespan


def _mount_receipt_routes(app: FastAPI) -> None:
    """Receipt-verify routes: single + batch, kept out of ``create_app``
    to keep its branch complexity under the repo's ruff cap."""

    @app.post(
        "/receipts/verify",
        response_model=ReceiptVerifyResponse,
        tags=["receipts"],
        operation_id="verify_receipt",
    )
    def verify_receipt(body: ReceiptVerifyRequest) -> ReceiptVerifyResponse:
        result = verify_receipt_payload(body.receipt, path=Path("<api>"))
        return ReceiptVerifyResponse(
            valid=result["valid"],
            path=result["path"],
            schema_tag=result["schema"],
            kind=result["kind"] if isinstance(result["kind"], str) else None,
            verdict=result["verdict"] if isinstance(result["verdict"], str) else None,
            digest_convention=result["digest_convention"],
            errors=list(result["errors"]),
            warnings=list(result["warnings"]),
        )

    @app.post(
        "/receipts/verify/batch",
        response_model=ReceiptVerifyBatchResponse,
        tags=["receipts"],
        operation_id="verify_receipts_batch",
    )
    def verify_receipts_batch(body: ReceiptVerifyBatchRequest) -> ReceiptVerifyBatchResponse:
        items: list[ReceiptVerifyBatchItem] = []
        for i, receipt in enumerate(body.receipts):
            try:
                result = verify_receipt_payload(receipt, path=Path("<api>"))
            except Exception as exc:  # verifier must fail item-local, never 500
                items.append(
                    ReceiptVerifyBatchItem(
                        index=i,
                        valid=False,
                        path="<api>",
                        schema_tag="",
                        kind=None,
                        verdict=None,
                        digest_convention=None,
                        errors=[str(exc)],
                        warnings=[],
                    )
                )
                continue
            items.append(
                ReceiptVerifyBatchItem(
                    index=i,
                    valid=result["valid"],
                    path=result["path"],
                    schema_tag=result["schema"],
                    kind=result["kind"] if isinstance(result["kind"], str) else None,
                    verdict=result["verdict"] if isinstance(result["verdict"], str) else None,
                    digest_convention=result["digest_convention"],
                    errors=list(result["errors"]),
                    warnings=list(result["warnings"]),
                )
            )
        verified = sum(1 for it in items if it.valid)
        return ReceiptVerifyBatchResponse(
            verified=verified,
            failed=len(items) - verified,
            results=items,
        )


def _mount_job_routes(
    app: FastAPI,
    lab: Harness,
    job_store: _JobStore,
    metrics: _Metrics,
    inflight: threading.BoundedSemaphore,
    jobs_executor: ThreadPoolExecutor,
) -> None:
    """Async job lifecycle routes: submit (202) / status / list / cancel.
    Submit is gated by drain + max_inflight; reads and cancel are
    control-plane and stay open under drain."""

    @app.post(
        "/harness/jobs",
        response_model=JobSubmitResponse,
        status_code=202,
        tags=["jobs"],
        operation_id="submit_job",
    )
    def submit_job(
        body: HarnessRunRequest,
        response: Response,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JobSubmitResponse:
        """Async run submission: work starts in the background, the caller
        polls ``GET /harness/jobs/{job_id}`` for the terminal record (also
        echoed as the ``Location`` header). Same drain/cap/idempotency
        contract as the sync route."""
        out = _submit_job(
            body,
            idempotency_key or body.idempotency_key,
            lab,
            job_store,
            metrics,
            inflight,
            jobs_executor,
        )
        response.headers["Location"] = f"/harness/jobs/{out.job_id}"
        return out

    @app.post(
        "/harness/jobs/batch",
        response_model=JobBatchResponse,
        status_code=202,
        tags=["jobs"],
        operation_id="submit_jobs_batch",
    )
    def submit_jobs_batch(body: JobBatchRequest) -> JobBatchResponse:
        """Fan-out submit: each item takes the same path as the single
        route — command validation, drain latch, inflight cap, and
        per-item ``idempotency_key`` dedup (headers carry no per-item
        keys, so the body field is the channel here). An item that fails
        lands in its own ``jobs[i]`` slot as ``{error, code}`` — the
        batch reports honest ``submitted``/``failed`` counts rather than
        turning one bad item into a whole-batch failure."""
        items: list[JobBatchItemResponse] = []
        submitted = 0
        for i, req in enumerate(body.jobs):
            try:
                resp = _submit_job(
                    req,
                    req.idempotency_key,
                    lab,
                    job_store,
                    metrics,
                    inflight,
                    jobs_executor,
                )
                items.append(
                    JobBatchItemResponse(
                        index=i,
                        job_id=resp.job_id,
                        status=resp.status,
                        replayed=resp.replayed,
                    )
                )
                submitted += 1
            except ApiError as exc:
                items.append(JobBatchItemResponse(index=i, error=str(exc.detail), code=exc.code))
        return JobBatchResponse(jobs=items, submitted=submitted, failed=len(body.jobs) - submitted)

    @app.get(
        "/harness/jobs", response_model=JobListResponse, tags=["jobs"], operation_id="list_jobs"
    )
    def list_jobs(
        status: _JobStatus | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> JobListResponse:
        """Job inventory, newest first: ``status`` filter + offset/limit
        paging; ``total`` is the filtered count before the page is cut."""
        jobs = job_store.list(status=status)
        return JobListResponse(jobs=jobs[offset : offset + limit], total=len(jobs))

    @app.get(
        "/harness/jobs/{job_id}",
        response_model=JobStatusResponse,
        tags=["jobs"],
        operation_id="get_job",
    )
    def job_status(job_id: str) -> JobStatusResponse:
        job = job_store.get(job_id)
        if job is None:
            raise ApiError(404, f"unknown job_id {job_id!r}")
        return job

    @app.delete(
        "/harness/jobs/{job_id}",
        response_model=JobStatusResponse,
        tags=["jobs"],
        operation_id="cancel_job",
    )
    def cancel_job(job_id: str) -> JobStatusResponse:
        """Cooperative cancel: a queued job flips to 'cancelled' and its
        executor slot frees on dequeue. Running and terminal jobs 409 —
        the subprocess runner has no mid-run kill handle."""
        job, outcome = job_store.cancel(job_id)
        if job is None:
            raise ApiError(404, f"unknown job_id {job_id!r}")
        if outcome != "cancelled":
            raise ApiError(409, f"job {job_id!r} is {outcome}")
        _deliver_job_callback(job)  # cancelled is terminal — fire the webhook
        return job

    @app.get(
        "/harness/jobs/{job_id}/events",
        tags=["jobs"],
        operation_id="stream_job_events",
    )
    def job_events(
        job_id: str,
        timeout_s: float = Query(default=600.0, ge=1.0, le=3600.0),
    ) -> StreamingResponse:
        """SSE job-status stream: the job record arrives as an
        ``event: job`` frame on every change until a terminal status,
        then the stream closes. ``: keepalive`` comment frames fire every
        ``sse_keepalive_s`` while the job is quiet; ``timeout_s`` bounds
        the stream — reconnect (and resume from the last frame) to keep
        watching. Same read path as ``GET /harness/jobs/{job_id}`` — a
        snapshot before the stream opens, so an unknown id 404s rather
        than hanging."""
        job = job_store.get(job_id)
        if job is None:
            raise ApiError(404, f"unknown job_id {job_id!r}")
        keepalive_s = float(getattr(app.state, "sse_keepalive_s", 15.0))

        def _frames() -> Iterator[str]:
            deadline = time.monotonic() + timeout_s
            last = ""
            next_keep = time.monotonic() + keepalive_s if keepalive_s > 0 else math.inf
            poll_s = 0.25 if keepalive_s <= 0 else min(0.25, keepalive_s)
            while True:
                cur = job_store.get(job_id)
                if cur is None:
                    return  # evicted from the bounded store mid-stream
                snap = cur.model_dump_json()
                if snap != last:
                    last = snap
                    yield f"event: job\ndata: {snap}\n\n"
                now = time.monotonic()
                if cur.status in _TERMINAL_JOB_STATUS or now >= deadline:
                    return
                if now >= next_keep:
                    yield ": keepalive\n\n"
                    next_keep = now + keepalive_s
                time.sleep(poll_s)

        return StreamingResponse(
            _frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


class _IdemStore[IdemT: BaseModel]:
    """Bounded LRU of ``Idempotency-Key`` -> stored response.

    Lets a client (or the HarnessClient, which mints a key per ``run``)
    retry a submission after a transport blip without double-executing
    the work. Read-only replays bypass the drain latch and the
    concurrency cap: the work already happened.
    """

    def __init__(self, max_entries: int) -> None:
        self._lock = threading.Lock()
        self._max = max_entries
        self._map: OrderedDict[str, tuple[str, IdemT]] = OrderedDict()

    def get(self, key: str) -> tuple[str, IdemT] | None:
        with self._lock:
            hit = self._map.get(key)
            if hit is not None:
                self._map.move_to_end(key)
            return hit

    def put(self, key: str, fingerprint: str, resp: IdemT) -> None:
        with self._lock:
            self._map[key] = (fingerprint, resp)
            self._map.move_to_end(key)
            while len(self._map) > self._max:
                self._map.popitem(last=False)


class JobListResponse(_Model):
    """Job inventory page: ``total`` is the filtered count before paging."""

    jobs: list[JobStatusResponse]
    total: int


_JOB_BATCH_MAX = 64


class JobBatchRequest(_Model):
    """Batch submit envelope — ``min_length``/``max_length`` fail closed:
    an empty batch and an over-cap batch are both 422s."""

    jobs: list[HarnessRunRequest] = Field(min_length=1, max_length=_JOB_BATCH_MAX)


class JobBatchItemResponse(_Model):
    """Per-item outcome — either the submit tuple or the ApiError the
    item failed with (``code`` carries the machine-stable class)."""

    index: int
    job_id: str | None = None
    status: _JobStatus | None = None
    replayed: bool | None = None
    error: str | None = None
    code: str | None = None


class JobBatchResponse(_Model):
    jobs: list[JobBatchItemResponse]
    submitted: int
    failed: int


class _JobStore:
    """Bounded store of async run jobs + their Idempotency-Key index.

    Job submissions take a worker slot at submit time (same drain/cap
    contract as the sync route) and release it when the job finishes, so
    the queue can never grow past the declared concurrency bound. The
    store is an LRU; evicting a job also drops its idempotency mapping.
    """

    def __init__(self, max_entries: int) -> None:
        self._lock = threading.Lock()
        self._max = max_entries
        self._jobs: OrderedDict[str, JobStatusResponse] = OrderedDict()
        self._keys: OrderedDict[str, tuple[str, str]] = OrderedDict()
        self._job_key: dict[str, str] = {}

    def get(self, job_id: str) -> JobStatusResponse | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, status: str | None = None) -> list[JobStatusResponse]:
        """Newest-first snapshot, optionally filtered by status."""
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.reverse()
        if status is not None:
            jobs = [j for j in jobs if j.status == status]
        return jobs

    def cancel(self, job_id: str) -> tuple[JobStatusResponse | None, str]:
        """Cooperative cancel: a 'queued' job flips to 'cancelled' (the
        worker frees its slot on dequeue without running). Running and
        terminal jobs report their status so the route can 409."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None, "missing"
            if job.status == "queued":
                job.status = "cancelled"
                job.finished_at = time.time()
                return job, "cancelled"
            return job, job.status

    def cancel_pending(self) -> builtins.list[JobStatusResponse]:
        """Shutdown path: every queued job flips to 'cancelled' — its
        future never starts, so it holds no slot; running jobs finish or
        die with the process. Returns the cancelled records so the caller
        can fire their webhooks."""
        with self._lock:
            out = [j for j in self._jobs.values() if j.status == "queued"]
            for job in out:
                job.status = "cancelled"
                job.finished_at = time.time()
            return out

    def get_key(self, key: str) -> tuple[str, str] | None:
        with self._lock:
            hit = self._keys.get(key)
            if hit is not None:
                self._keys.move_to_end(key)
            return hit

    def put(
        self,
        job: JobStatusResponse,
        key: str | None,
        fingerprint: str | None,
    ) -> None:
        with self._lock:
            self._jobs[job.job_id] = job
            self._jobs.move_to_end(job.job_id)
            if key is not None and fingerprint is not None:
                self._keys[key] = (fingerprint, job.job_id)
                self._keys.move_to_end(key)
                self._job_key[job.job_id] = key
            while len(self._jobs) > self._max:
                old_id, _ = self._jobs.popitem(last=False)
                old_key = self._job_key.pop(old_id, None)
                if old_key is not None:
                    self._keys.pop(old_key, None)


def _backend_configured() -> dict[str, bool]:
    """Presence-of-credentials flags only — values never leave the process."""
    checkpoint_env = os.environ.get("FX1_CHECKPOINT_DIR", "")
    return {
        "hosted_k3": bool(os.environ.get("MOONSHOT_API_KEY")),
        "byok": all(
            os.environ.get(name) for name in (BYOK_BASE_URL_ENV, BYOK_API_KEY_ENV, BYOK_MODEL_ENV)
        ),
        "local_fx1": bool(checkpoint_env)
        and (Path(checkpoint_env) / "modelcard.json").is_file()
        and bool(os.environ.get(LOCAL_SERVE_URL_ENV) or os.environ.get(LOCAL_SERVE_CMD_ENV)),
    }


_RID_OK = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _request_id(raw: str | None) -> str:
    """Echo a well-formed client request id; anything else mints a fresh one
    (untrusted headers never reach the response unparsed)."""
    if raw is not None and _RID_OK.fullmatch(raw):
        return raw
    return uuid.uuid4().hex


logger = logging.getLogger("fx1.serve.api")
if not logger.handlers:  # embedders may still attach their own handlers
    _log_handler = logging.StreamHandler()
    _log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s fx1-api %(message)s"))
    logger.addHandler(_log_handler)
    logger.setLevel(logging.INFO)


def _finish(request: Request, request_id: str, response: Any, started: float) -> Any:
    """Single post-processing tail for every response — security headers,
    request-id echo, and one structured access line."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Fx1-Api-Version"] = API_VERSION
    request.app.state.metrics.record(response.status_code)
    logger.info(
        "request method=%s path=%s status=%d elapsed_ms=%.1f rid=%s",
        request.method,
        request.url.path,
        response.status_code,
        (time.monotonic() - started) * 1000,
        request_id,
    )
    return response


class _RateLimiter:
    """Per-client-host token bucket: ``rps`` refill with one second of
    burst capacity, LRU-bounded so a spray of source addresses can't grow
    the map without limit. `allow` returns 0.0 when the request may
    proceed, else the seconds until a token refills."""

    def __init__(self, rps: float, max_keys: int = _RATE_LIMIT_KEYS_MAX) -> None:
        self.rps = rps
        self.capacity = max(1.0, rps)
        self.max_keys = max_keys
        self._lock = threading.Lock()
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()

    def allow(self, identity: str) -> tuple[float, float]:
        """Consume one token: returns ``(wait, remaining)`` — seconds until a
        token refills (0.0 when the request may proceed) and the post-consume
        token count for the `X-RateLimit-Remaining` header."""
        now = time.monotonic()
        with self._lock:
            entry = self._buckets.pop(identity, None)
            tokens, ts = entry if entry is not None else (self.capacity, now)
            tokens = min(self.capacity, tokens + self.rps * (now - ts))
            if tokens >= 1.0:
                self._buckets[identity] = (tokens - 1.0, now)
                return 0.0, max(0.0, tokens - 1.0)
            wait = (1.0 - tokens) / self.rps
            self._buckets[identity] = (tokens, now)
            while len(self._buckets) > self.max_keys:
                self._buckets.popitem(last=False)
            return wait, 0.0


class _BackendBreaker:
    """Consecutive-fault circuit breaker keyed on backend name.

    `threshold` consecutive call faults open the circuit for
    `cooldown_s`: calls fail fast (503 + Retry-After) without touching the
    backend or holding an inflight slot. When the window lapses a single
    half-open probe is admitted; success closes the circuit, failure
    re-opens it for a fresh window. A probe in flight fast-fails other
    callers — only the probe touches the backend. Configuration faults
    (501, honesty-gate refusals) never count: the breaker measures
    backend health, not model output.
    """

    def __init__(self, threshold: int, cooldown_s: float) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._lock = threading.Lock()
        self._fails: dict[str, int] = {}
        self._opened: dict[str, float] = {}
        self._probing: set[str] = set()

    def check(self, backend: str) -> float:
        """Admit or refuse a call: 0.0 proceeds, >0 fast-fails for that
        many seconds of remaining cooldown."""
        with self._lock:
            opened = self._opened.get(backend)
            if opened is None:
                return 0.0
            remaining = self.cooldown_s - (time.monotonic() - opened)
            if remaining <= 0.0:
                if backend in self._probing:
                    return 0.01
                self._probing.add(backend)
                return 0.0
            return remaining

    def report(self, backend: str, ok: bool) -> None:
        """Record the admitted call's outcome."""
        with self._lock:
            self._probing.discard(backend)
            if ok:
                self._fails.pop(backend, None)
                self._opened.pop(backend, None)
                return
            fails = self._fails.get(backend, 0) + 1
            self._fails[backend] = fails
            if fails >= self.threshold:
                self._opened[backend] = time.monotonic()

    def state(self, backend: str) -> tuple[bool, float, int]:
        """``(circuit_open, cooldown_remaining_s, consecutive_failures)``."""
        with self._lock:
            opened = self._opened.get(backend)
            remaining = (
                max(0.0, self.cooldown_s - (time.monotonic() - opened))
                if opened is not None
                else 0.0
            )
            return opened is not None and remaining > 0.0, remaining, self._fails.get(backend, 0)


def _render_prometheus(snap: MetricsResponse, job_counts: dict[str, int]) -> str:
    """Render the ops snapshot as Prometheus text exposition (format
    0.0.4) — ``GET /metrics`` stays JSON by default; ``?format=prom``
    or an ``Accept: text/plain``/OpenMetrics header selects this view.
    Label values come from integer status codes and the fixed
    ``_JOB_STATUSES`` set, so quoting is mechanical, not user data."""

    def _label(v: str) -> str:
        return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    lines = [
        "# HELP fx1_uptime_seconds Seconds since the harness API process started.",
        "# TYPE fx1_uptime_seconds gauge",
        f"fx1_uptime_seconds {snap.uptime_s:.6f}",
        "# HELP fx1_requests_total Harness API requests by response status code.",
        "# TYPE fx1_requests_total counter",
    ]
    for status in sorted(snap.by_status):
        lines.append(f'fx1_requests_total{{status="{_label(status)}"}} {snap.by_status[status]}')
    lines += [
        "# HELP fx1_errors_total Harness API responses with status >= 400.",
        "# TYPE fx1_errors_total counter",
        f"fx1_errors_total {snap.errors_total}",
        "# HELP fx1_rate_limited_total Requests denied by the rate limiter.",
        "# TYPE fx1_rate_limited_total counter",
        f"fx1_rate_limited_total {snap.rate_limited_total}",
        "# HELP fx1_inflight Requests currently executing.",
        "# TYPE fx1_inflight gauge",
        f"fx1_inflight {snap.inflight}",
        "# HELP fx1_inflight_watermark High-water mark of concurrent requests.",
        "# TYPE fx1_inflight_watermark gauge",
        f"fx1_inflight_watermark {snap.inflight_watermark}",
        "# HELP fx1_max_inflight Configured concurrency bound.",
        "# TYPE fx1_max_inflight gauge",
        f"fx1_max_inflight {snap.max_inflight}",
        "# HELP fx1_draining 1 while the drain latch is set.",
        "# TYPE fx1_draining gauge",
        f"fx1_draining {1 if snap.draining else 0}",
        "# HELP fx1_jobs Harness jobs by lifecycle status.",
        "# TYPE fx1_jobs gauge",
    ]
    for status in _JOB_STATUSES:
        lines.append(f'fx1_jobs{{status="{status}"}} {job_counts.get(status, 0)}')
    return "\n".join(lines) + "\n"


def _close_backend(backend: Any) -> None:
    closer = getattr(backend, "close", None)
    if callable(closer):
        closer()


def _mount_complete_routes(
    app: FastAPI,
    *,
    slot: Callable[[], Iterator[None]],
    resolve_backend: Callable[[str, str | None], Any],
    sse_keepalive_s: float,
    complete_idem_store: _IdemStore[CompleteResponse],
    complete_batch_idem_store: _IdemStore[CompleteBatchResponse],
    breaker: _BackendBreaker | None,
) -> None:
    """Complete routes (sync / SSE stream / batch) — extracted from
    ``create_app`` to keep its branch complexity under the ruff cap."""

    def _breaker_admit(name: str) -> None:
        """Fast-fail while the backend's circuit is open — the call never
        burns an inflight slot waiting on a dead endpoint."""
        if breaker is None:
            return
        wait = breaker.check(name)
        if wait > 0:
            raise ApiError(
                503,
                f"backend {name!r} circuit open — retry in {wait:.1f}s",
                code="backend_unavailable",
                headers={"Retry-After": str(max(1, math.ceil(wait)))},
            )

    @app.post(
        "/harness/complete",
        response_model=CompleteResponse,
        tags=["complete"],
        operation_id="complete",
    )
    def complete(
        body: CompleteRequest,
        _slot_held: None = Depends(slot),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> CompleteResponse:
        # Same Idempotency-Key contract as the runs route: retried
        # completions replay from the cache instead of re-billing the model.
        body_fp = body.model_dump_json()
        key, replay = _idem_lookup(idempotency_key, complete_idem_store, body_fp)
        if replay is not None:
            return replay
        _breaker_admit(body.backend)
        try:
            backend = resolve_backend(body.backend, body.checkpoint_dir)
        except ApiError as exc:
            # only backend-unavailable counts — client errors (404 unknown
            # backend, 422 bad args) must never trip the circuit, or a caller
            # could deny the backend for everyone by spamming bad requests.
            if breaker is not None and exc.status_code == 503:
                breaker.report(body.backend, False)
            raise
        messages = [{"role": m.role, "content": m.content} for m in body.messages]
        try:
            content = cited_complete(backend, messages, receipt_hashes=body.receipt_hashes)
        except NotImplementedError as exc:
            raise ApiError(501, str(exc)) from exc
        except Fx1HonestyError as exc:
            # The model produced a contract-violating headline; the gate
            # caught it before the bytes left — surface as 502, not success.
            raise ApiError(
                502, f"honesty gate refused model output: {exc}", code="honesty_gate"
            ) from exc
        except (BackendNotConfiguredError, RuntimeError) as exc:
            if breaker is not None:
                breaker.report(body.backend, False)
            if isinstance(exc, BackendNotConfiguredError):
                raise ApiError(503, str(exc), code="backend_unavailable") from exc
            raise ApiError(502, str(exc), code="backend_failure") from exc
        else:
            if breaker is not None:
                breaker.report(body.backend, True)
        finally:
            _close_backend(backend)
        model_name = getattr(backend, "_model", None)
        resp = CompleteResponse(
            backend=body.backend,
            model=model_name if isinstance(model_name, str) else None,
            content=content,
            receipt_hashes=body.receipt_hashes or [],
        )
        if key is not None:
            complete_idem_store.put(key, body_fp, resp)
        return resp

    @app.post(
        "/harness/complete/stream",
        tags=["complete"],
        operation_id="complete_stream",
    )
    def complete_stream(
        body: CompleteRequest, _slot_held: None = Depends(slot)
    ) -> StreamingResponse:
        """Server-sent-event stream of one gated completion.

        The backend's token deltas are buffered, the joined text passes the
        honesty gate, and only then are chunks emitted as ``token`` events
        plus a ``final`` envelope — no ungated model bytes ever reach a
        ``data:`` frame. A backend without ``stream`` fails 501 rather than
        faking chunking.

        With ``sse_keepalive_s`` > 0 (default 15) the buffer+gate phase runs
        on a worker thread under a one-interval grace period: outcomes that
        resolve within the interval keep today's wire contract exactly —
        completions stream as usual, failures stay ordinary JSON error
        responses. Only a generation still running past the interval
        commits to SSE: ``: keepalive`` comment frames (no payload —
        parsers ignore them) hold the connection open against proxy/LB
        idle timeouts, and a failure after that point arrives as a terminal
        ``{"type": "error", "status", "detail"}`` data frame followed by
        ``[DONE]`` (the HTTP status is committed once a byte is on the
        wire). ``sse_keepalive_s`` = 0 disables the worker entirely — the
        fully synchronous path.
        """
        messages = [{"role": m.role, "content": m.content} for m in body.messages]

        def _gather() -> tuple[list[str], str | None]:
            """Buffer + gate the backend stream; raises the mapped errors."""
            _breaker_admit(body.backend)
            try:
                backend = resolve_backend(body.backend, body.checkpoint_dir)
            except ApiError as exc:
                if breaker is not None and exc.status_code == 503:
                    breaker.report(body.backend, False)
                raise
            try:
                if not isinstance(backend, StreamingBackend):
                    raise NotImplementedError(
                        f"backend {body.backend!r} does not support streaming"
                    )
                chunks = list(backend.stream(messages))
                joined = "".join(chunks)
                try:
                    validate_fx1_output(joined)
                except Fx1HonestyError as exc:
                    raise ApiError(
                        502, f"honesty gate refused model output: {exc}", code="honesty_gate"
                    ) from exc
            except NotImplementedError as exc:
                raise ApiError(501, str(exc)) from exc
            except (BackendNotConfiguredError, RuntimeError) as exc:
                if breaker is not None:
                    breaker.report(body.backend, False)
                if isinstance(exc, BackendNotConfiguredError):
                    raise ApiError(503, str(exc), code="backend_unavailable") from exc
                raise ApiError(502, str(exc), code="backend_failure") from exc
            else:
                if breaker is not None:
                    breaker.report(body.backend, True)
            finally:
                _close_backend(backend)
            model_name = getattr(backend, "_model", None)
            if body.receipt_hashes:
                chunks.append(
                    "\n\nEvidence: "
                    + ", ".join(f"`{h[:16]}…`" for h in body.receipt_hashes)
                    + " — verify with `dipcatcher verify-research`."
                )
            return chunks, model_name if isinstance(model_name, str) else None

        def _events(chunks: list[str], model_name: str | None) -> Iterator[str]:
            for chunk in chunks:
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "final",
                        "model": model_name,
                        "receipt_hashes": body.receipt_hashes or [],
                    }
                )
                + "\n\n"
            )
            yield "data: [DONE]\n\n"

        if sse_keepalive_s <= 0:
            chunks, model_name = _gather()
            return StreamingResponse(_events(chunks, model_name), media_type="text/event-stream")

        pipe: queue.Queue[tuple[str, Any]] = queue.Queue()

        def _produce() -> None:
            try:
                pipe.put(("ok", _gather()))
            except HTTPException as exc:
                pipe.put(("error", exc))
            except Exception as exc:  # noqa: BLE001 — dead pipe, honest frame
                pipe.put(("error", HTTPException(502, f"backend failed: {exc}")))

        threading.Thread(target=_produce, daemon=True).start()

        # Grace window: an outcome inside one keepalive interval keeps the
        # synchronous contract (SSE stream / JSON error); only a generation
        # still running past it commits to the keepalived stream.
        grace: tuple[str, Any] | None
        try:
            grace = pipe.get(timeout=sse_keepalive_s)
        except queue.Empty:
            grace = None
        if grace is not None:
            tag, payload = grace
            if tag == "error":
                raise payload
            chunks, model_name = payload
            return StreamingResponse(_events(chunks, model_name), media_type="text/event-stream")

        def _events_keepalived() -> Iterator[str]:
            while True:
                try:
                    tag, payload = pipe.get(timeout=sse_keepalive_s)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                if tag == "error":
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "type": "error",
                                "status": payload.status_code,
                                "detail": payload.detail,
                                "code": _err_code(payload),
                            }
                        )
                        + "\n\n"
                    )
                    yield "data: [DONE]\n\n"
                    return
                chunks, model_name = payload
                yield from _events(chunks, model_name)
                return

        return StreamingResponse(_events_keepalived(), media_type="text/event-stream")

    @app.post(
        "/harness/complete/batch",
        response_model=CompleteBatchResponse,
        tags=["complete"],
        operation_id="complete_batch",
    )
    def complete_batch(
        body: CompleteBatchRequest,
        _slot_held: None = Depends(slot),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> CompleteBatchResponse:
        body_fp = body.model_dump_json()
        key, replay = _idem_lookup(idempotency_key, complete_batch_idem_store, body_fp)
        if replay is not None:
            return replay
        _breaker_admit(body.backend)
        try:
            backend = resolve_backend(body.backend, body.checkpoint_dir)
        except ApiError as exc:
            # only backend-unavailable counts — client errors (404 unknown
            # backend, 422 bad args) must never trip the circuit, or a caller
            # could deny the backend for everyone by spamming bad requests.
            if breaker is not None and exc.status_code == 503:
                breaker.report(body.backend, False)
            raise
        # One backend serves the whole batch — a spawned local engine is
        # shared across workers (spawn path is lock-guarded). Item failures
        # are per-slot verdicts: a gate refusal on one prompt does not lose
        # the rest of the batch. An open circuit short-circuits items
        # individually so a batch over a dead backend ends fast.
        try:
            with ThreadPoolExecutor(
                max_workers=min(body.max_workers, len(body.batch)),
                thread_name_prefix="fx1-complete",
            ) as pool:

                def _one(messages: list[dict[str, str]]) -> CompleteBatchItem:
                    if breaker is not None and breaker.check(body.backend) > 0:
                        return CompleteBatchItem(
                            ok=False,
                            error="backend circuit open",
                            error_class="backend_unavailable",
                        )
                    try:
                        out = CompleteBatchItem(
                            ok=True,
                            content=cited_complete(
                                backend, messages, receipt_hashes=body.receipt_hashes
                            ),
                        )
                    except Fx1HonestyError as exc:
                        return CompleteBatchItem(
                            ok=False, error=str(exc), error_class="honesty_refusal"
                        )
                    except (
                        BackendNotConfiguredError,
                        RuntimeError,
                        ValueError,
                    ) as exc:
                        if breaker is not None and not isinstance(exc, NotImplementedError):
                            breaker.report(body.backend, False)
                        return CompleteBatchItem(
                            ok=False, error=str(exc), error_class=type(exc).__name__
                        )
                    if breaker is not None:
                        breaker.report(body.backend, True)
                    return out

                results = list(
                    pool.map(
                        _one,
                        [
                            [{"role": m.role, "content": m.content} for m in msgs]
                            for msgs in body.batch
                        ],
                    )
                )
        finally:
            _close_backend(backend)
        model_name = getattr(backend, "_model", None)
        resp = CompleteBatchResponse(
            backend=body.backend,
            model=model_name if isinstance(model_name, str) else None,
            receipt_hashes=body.receipt_hashes or [],
            results=results,
        )
        if key is not None:
            complete_batch_idem_store.put(key, body_fp, resp)
        return resp


def create_app(
    harness: Harness | None = None,
    backend_resolver: Any | None = None,
    max_inflight: int | None = None,
    sse_keepalive_s: float | None = None,
    idem_max: int | None = None,
    job_max: int | None = None,
    rate_limit_rps: float | None = None,
    gzip_min_bytes: int | None = None,
    cors_origins: str | None = None,
    breaker_threshold: int | None = None,
    breaker_cooldown_s: float | None = None,
) -> FastAPI:
    api_key = os.environ.get(_API_KEY_ENV) or None
    lab = harness or Harness()
    resolve_backend = backend_resolver or get_backend
    max_inflight = _env_int_bound(_MAX_INFLIGHT_ENV, 16, max_inflight)
    idem_max = _env_int_bound(_IDEM_MAX_ENV, 1024, idem_max)
    job_max = _env_int_bound(_JOB_MAX_ENV, 1024, job_max)
    sse_keepalive_s = _env_float_floor(_SSE_KEEPALIVE_ENV, 15.0, sse_keepalive_s)
    rate_limit_rps = _env_float_floor(_RATE_LIMIT_ENV, 0.0, rate_limit_rps)
    gzip_min_bytes = _env_int_floor(_GZIP_MIN_ENV, 1024, gzip_min_bytes)
    breaker_threshold = _env_int_floor(_BREAKER_THRESHOLD_ENV, 5, breaker_threshold)
    breaker_cooldown_s = _env_float_floor(_BREAKER_COOLDOWN_ENV, 30.0, breaker_cooldown_s)
    cors_raw = cors_origins if cors_origins is not None else os.environ.get(_CORS_ORIGINS_ENV, "")
    cors_list = [o.strip() for o in cors_raw.split(",") if o.strip()]
    for origin in cors_list:
        parsed = urllib.parse.urlparse(origin)
        if origin == "*" or parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                f"invalid CORS origin {origin!r} — expected explicit scheme://host, "
                "never the wildcard"
            )
    limiter = _RateLimiter(rate_limit_rps) if rate_limit_rps > 0 else None
    breaker = (
        _BackendBreaker(breaker_threshold, breaker_cooldown_s) if breaker_threshold > 0 else None
    )
    # Bounded in-flight work: the harness executes lab commands and model
    # calls on shared resources (a spawned local engine, GPU memory, the
    # box itself) — saturation must fail honestly as 503, never queue
    # unboundedly or crash mid-request. Cheap routes (commands, verify,
    # health) stay uncapped so liveness answers under load.
    inflight = threading.BoundedSemaphore(max_inflight)
    metrics = _Metrics(max_inflight)
    idem_store: _IdemStore[HarnessRunResponse] = _IdemStore(idem_max)
    complete_idem_store: _IdemStore[CompleteResponse] = _IdemStore(idem_max)
    complete_batch_idem_store: _IdemStore[CompleteBatchResponse] = _IdemStore(idem_max)
    job_store = _JobStore(job_max)
    jobs_executor = ThreadPoolExecutor(max_workers=max_inflight, thread_name_prefix="fx1-job")

    @contextmanager
    def _work_gate() -> Iterator[None]:
        if metrics.draining.is_set():
            raise ApiError(
                503,
                "harness is draining — no new work accepted",
                code="draining",
            )
        if not inflight.acquire(blocking=False):
            raise ApiError(
                503,
                f"harness at max_inflight={max_inflight} — retry later",
                code="over_capacity",
                headers={"Retry-After": "1"},
            )
        metrics.acquire()
        try:
            yield
        finally:
            metrics.release()
            inflight.release()

    def _slot() -> Iterator[None]:
        with _work_gate():
            yield

    app = FastAPI(
        title="fx-1 harness API",
        version=__version__,
        description=(
            "Execution surface for the fx-1 harness: the registered lab "
            "commands, sealed-receipt verification, and gated model "
            "completion over hosted_k3 / local_fx1 / BYOK backends."
        ),
        lifespan=_make_lifespan(metrics, job_store, jobs_executor),
        openapi_tags=[
            {"name": "runs", "description": "Synchronous lab-command execution."},
            {"name": "jobs", "description": "Async run jobs: submit, poll, SSE, cancel, batch."},
            {
                "name": "complete",
                "description": "Gated model completion (sync, batch, SSE stream).",
            },
            {"name": "receipts", "description": "Sealed-receipt verification."},
            {"name": "ops", "description": "Liveness, readiness, metrics, drain, version."},
        ],
    )
    # Large responses (job listings, receipt payloads, openapi) compress well;
    # urllib-based clients send no Accept-Encoding so SSE stays uncompressed.
    if gzip_min_bytes > 0:
        app.add_middleware(GZipMiddleware, minimum_size=gzip_min_bytes)
    app.state.cors_origins = cors_list
    app.state.gzip_min_bytes = gzip_min_bytes
    app.state.inflight_slots = inflight
    app.state.metrics = metrics
    app.state.idem_store = idem_store
    app.state.job_store = job_store
    app.state.jobs_executor = jobs_executor
    app.state.sse_keepalive_s = sse_keepalive_s
    app.state.rate_limiter = limiter
    app.state.breaker = breaker

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": _err_code(exc)},
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder({"detail": exc.errors(), "code": "validation"}),
        )

    @app.middleware("http")
    async def harness_api_auth(request: Request, call_next: Any) -> Any:
        request_id = _request_id(request.headers.get("x-request-id"))
        request.state.request_id = request_id
        started = time.monotonic()
        rl_headers: dict[str, str] | None = None
        if limiter is not None:
            # keyed on client host so a rotating fake API key can't evade it
            host = (request.client.host if request.client else "") or "unknown"
            wait, remaining = limiter.allow(host)
            reset = max(0, math.ceil((limiter.capacity - remaining) / limiter.rps))
            rl_headers = {
                "X-RateLimit-Limit": str(int(limiter.rps)),
                "X-RateLimit-Remaining": str(int(remaining)),
                "X-RateLimit-Reset": str(reset),
            }
            if wait > 0:
                metrics.record_rate_limited()
                response = JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"rate limit exceeded; retry in {wait:.1f}s",
                        "code": "too_many_requests",
                    },
                    headers={"Retry-After": str(max(1, math.ceil(wait))), **rl_headers},
                )
                return _finish(request, request_id, response, started)
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            declared = request.headers.get("content-length")
            if declared is not None:
                try:
                    length = int(declared)
                except ValueError:
                    response = JSONResponse(
                        status_code=400,
                        content={"detail": "invalid content-length", "code": "bad_request"},
                    )
                    return _finish(request, request_id, response, started)
                if length > _MAX_BODY_BYTES:
                    response = JSONResponse(
                        status_code=413,
                        content={
                            "detail": f"body exceeds {_MAX_BODY_BYTES}-byte cap",
                            "code": "too_large",
                        },
                    )
                    return _finish(request, request_id, response, started)
        if request.url.path in _PUBLIC_PATHS:
            response = await call_next(request)
        elif api_key:
            provided = request.headers.get("X-API-Key")
            if not provided or not hmac.compare_digest(provided, api_key):
                response = JSONResponse(
                    status_code=401,
                    content={"detail": "invalid or missing X-API-Key", "code": "unauthorized"},
                )
            else:
                response = await call_next(request)
        else:
            host = (request.client.host if request.client else "") or ""
            if host not in _LOOPBACK_HOSTS:
                response = JSONResponse(
                    status_code=403,
                    content={
                        "detail": (
                            "FX1_API_KEY is unset; non-localhost clients are "
                            "refused. Set FX1_API_KEY and send X-API-Key, or "
                            "bind to 127.0.0.1 only."
                        ),
                        "code": "forbidden",
                    },
                )
            else:
                response = await call_next(request)
        if rl_headers is not None:
            response.headers.update(rl_headers)
        return _finish(request, request_id, response, started)

    @app.get("/metrics", response_model=MetricsResponse, tags=["ops"], operation_id="get_metrics")
    def metrics_route(
        request: Request,
        format: Literal["json", "prom", "prometheus"] | None = None,
    ) -> MetricsResponse | PlainTextResponse:
        """Ops snapshot — JSON by default; Prometheus text exposition via
        ``?format=prom`` or an ``Accept: text/plain``/OpenMetrics header."""
        accept = request.headers.get("accept", "")
        wants_text = format in ("prom", "prometheus") or (
            format is None and ("text/plain" in accept or "openmetrics" in accept)
        )
        if wants_text:
            snap = metrics.snapshot()
            counts: dict[str, int] = {}
            for job in job_store.list():
                counts[job.status] = counts.get(job.status, 0) + 1
            return PlainTextResponse(
                _render_prometheus(snap, counts),
                media_type="text/plain; version=0.0.4",
            )
        return metrics.snapshot()

    @app.get("/health", response_model=HealthResponse, tags=["ops"], operation_id="health")
    def health() -> HealthResponse:
        return HealthResponse(
            registered_commands=len(lab.list_commands()),
            backends=_backend_configured(),
            draining=metrics.draining.is_set(),
        )

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"], operation_id="ready")
    def ready() -> ReadyResponse:
        """Kubernetes-style readiness: 200 while accepting work, 503 once
        drain is latched — the load balancer's signal to deregister the
        pod before gated routes start refusing."""
        if metrics.draining.is_set():
            raise ApiError(503, "harness is draining", code="draining")
        return ReadyResponse(inflight=metrics.snapshot().inflight)

    app.add_api_route(
        "/harness/version",
        _version_info,
        methods=["GET"],
        response_model=VersionResponse,
        tags=["ops"],
        operation_id="get_version",
    )

    @app.get(
        "/harness/capabilities",
        response_model=CapabilitiesResponse,
        tags=["ops"],
        operation_id="get_capabilities",
    )
    def capabilities() -> CapabilitiesResponse:
        """Discovery: the wire features this build serves and the
        operational limits actually in effect. A client learns batch
        caps, store bounds, rate limits, and feature flags (idempotency,
        SSE, webhooks, drain) from one call — nothing is hardcoded."""
        return CapabilitiesResponse(
            api_version=API_VERSION,
            fx1_version=__version__,
            features={
                "idempotency": True,
                "sse": True,
                "webhooks": True,
                "batch": True,
                "jobs": True,
                "drain": True,
                "streaming": True,
                "cors": bool(cors_list),
                "breaker": breaker is not None,
            },
            limits={
                "max_inflight": float(metrics.max_inflight),
                "job_max": float(job_store._max),
                "idem_max": float(idem_store._max),
                "job_batch_max": float(_JOB_BATCH_MAX),
                "verify_batch_max": float(_VERIFY_BATCH_MAX),
                "complete_batch_max": 64.0,
                "body_max_bytes": float(_MAX_BODY_BYTES),
                "job_result_max_bytes": float(_JOB_RESULT_MAX_BYTES),
                "rate_limit_rps": limiter.rps if limiter is not None else 0.0,
                "breaker_threshold": float(breaker_threshold),
                "breaker_cooldown_s": float(breaker_cooldown_s),
                "sse_keepalive_s": float(sse_keepalive_s),
                "gzip_min_bytes": float(gzip_min_bytes),
            },
            backends=_backend_configured(),
            roles=sorted({str(c.role) for c in lab.list_commands()}),
        )

    @app.get(
        "/harness/backends",
        response_model=dict[str, BackendStatusEntry],
        tags=["ops"],
        operation_id="backends_status",
    )
    def backends_status() -> dict[str, BackendStatusEntry]:
        """Per-backend liveness: configured flag plus circuit state —
        whether the breaker is fast-failing this backend, how much
        cooldown remains, and the consecutive-fault streak."""
        configured = _backend_configured()
        out: dict[str, BackendStatusEntry] = {}
        for name, cfg in configured.items():
            circuit_open, remaining, fails = (
                breaker.state(name) if breaker is not None else (False, 0.0, 0)
            )
            out[name] = BackendStatusEntry(
                configured=cfg,
                circuit_open=circuit_open,
                cooldown_remaining_s=round(remaining, 3),
                consecutive_failures=fails,
            )
        return out

    @app.post(
        "/harness/drain",
        response_model=DrainResponse,
        tags=["ops"],
        operation_id="drain",
    )
    def drain(
        wait_s: float = Query(default=0.0, ge=0.0, le=600.0),
    ) -> DrainResponse:
        """Latch drain mode (one-way): gated routes refuse new work with
        503 while in-flight requests finish; ``/health``, ``/metrics``
        and ``/ready`` keep answering so orchestrators can watch
        ``inflight`` bleed to zero before stopping the process.
        Idempotent — re-POSTing just re-reads the latch. ``wait_s>0``
        blocks (server-side) until in-flight work empties or the window
        lapses; ``drained`` reports which happened."""
        metrics.draining.set()
        if wait_s > 0:
            metrics.wait_idle(wait_s)
        snap = metrics.snapshot()
        return DrainResponse(draining=True, inflight=snap.inflight, drained=snap.inflight == 0)

    @app.get(
        "/harness/commands",
        response_model=HarnessCommandListResponse,
        tags=["runs"],
        operation_id="list_commands",
    )
    def list_commands(
        role: HarnessRole | None = Query(default=None),
    ) -> HarnessCommandListResponse:
        items = [
            HarnessCommandItem(
                name=c.name,
                role=str(c.role),
                argv=list(c.argv),
                timeout_s=c.timeout_s,
                description=c.description,
            )
            for c in lab.list_commands(role)
        ]
        return HarnessCommandListResponse(items=items, total=len(items))

    @app.post(
        "/harness/runs",
        response_model=HarnessRunResponse,
        tags=["runs"],
        operation_id="run_command",
    )
    def run_command(
        body: HarnessRunRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> HarnessRunResponse:
        body_fp = body.model_dump_json(exclude={"idempotency_key"})
        key, replay = _idem_lookup(idempotency_key or body.idempotency_key, idem_store, body_fp)
        if replay is not None:
            return replay
        with _work_gate():
            try:
                result = lab.run(
                    body.command,
                    body.extra_args or None,
                    config=Path(body.config) if body.config else None,
                )
            except KeyError as exc:
                raise ApiError(404, str(exc)) from exc
            except ValueError as exc:
                raise ApiError(422, str(exc)) from exc
            command = lab.get(body.command)
            resp = HarnessRunResponse(
                command=result.command,
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                ok=result.ok,
                timeout_s=command.timeout_s,
            )
        if key is not None:
            idem_store.put(key, body_fp, resp)
        return resp

    _mount_job_routes(app, lab, job_store, metrics, inflight, jobs_executor)

    def _resolve_request_backend(backend_name: str, checkpoint_dir: str | None) -> Any:
        """Checkpoint validation + backend resolution → HTTP error map."""
        kwargs: dict[str, Any] = {}
        if backend_name == "local_fx1":
            checkpoint = checkpoint_dir or os.environ.get("FX1_CHECKPOINT_DIR")
            if not checkpoint:
                raise ApiError(
                    422,
                    "local_fx1 needs a checkpoint_dir in the request or "
                    "FX1_CHECKPOINT_DIR on the server",
                )
            kwargs["checkpoint_dir"] = checkpoint
        elif checkpoint_dir is not None:
            raise ApiError(422, "checkpoint_dir applies only to the local_fx1 backend")
        try:
            return resolve_backend(backend_name, **kwargs)
        except KeyError as exc:
            raise ApiError(404, str(exc)) from exc
        except FileNotFoundError as exc:
            raise ApiError(422, str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            # Missing credentials / unsigned release / failed ship gate are
            # server-side configuration faults, not client input errors.
            raise ApiError(503, str(exc), code="backend_unavailable") from exc

    _mount_complete_routes(
        app,
        slot=_slot,
        resolve_backend=_resolve_request_backend,
        sse_keepalive_s=sse_keepalive_s,
        complete_idem_store=complete_idem_store,
        complete_batch_idem_store=complete_batch_idem_store,
        breaker=breaker,
    )

    _mount_receipt_routes(app)

    # Opt-in CORS for browser consumers: off by default (closed), explicit
    # origins only — the wildcard and credentials are refused. Registered
    # last so it wraps the auth middleware — preflight OPTIONS reach CORS
    # before the API-key check (preflights carry no credentials).
    if cors_list:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_list,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=_CORS_ALLOW_HEADERS,
            expose_headers=_CORS_EXPOSE_HEADERS,
            allow_credentials=False,
            max_age=600,
        )

    _declare_response_headers(app, rate_limited=limiter is not None)

    return app


app = create_app()


if __name__ == "__main__":
    # `python -m fx1.serve.api` — loopback guard matches the research API.
    import uvicorn

    host = os.environ.get("FX1_API_HOST", "127.0.0.1")
    port = int(os.environ.get("FX1_API_PORT", "8011"))
    if host not in {"127.0.0.1", "::1", "localhost"} and not os.environ.get(_API_KEY_ENV):
        raise SystemExit(
            "non-loopback binding requires FX1_API_KEY; refusing unauthenticated exposure"
        )
    uvicorn.run(app, host=host, port=port, reload=False, server_header=False)
