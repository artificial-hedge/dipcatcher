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
import hashlib
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
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import (
    AbstractAsyncContextManager,
    asynccontextmanager,
    contextmanager,
)
from pathlib import Path
from typing import Any, Literal, cast

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
    SamplingParams,
    StreamingBackend,
    get_backend,
)
from fx1.serve.chat import cited_complete
from fx1.serve.contract import API_VERSION
from fx1.serve.evals import (
    EVAL_SAMPLING,
    EVAL_SUITES,
    EvalRecord,
    EvalStore,
    EvalSuiteName,
    eval_record_receipt,
    metered_model,
    run_eval_record,
    suite_accepts_judge,
)
from fx1.serve.openai_compat import (
    OPENAI_MODEL_IDS,
    ByokOverride,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAICompatError,
    OpenAIModel,
    OpenAIModelList,
    is_openai_path,
    openai_chunks,
    openai_envelope,
    openai_error_body,
    openai_to_kwargs,
)
from fx1.serve.receipt_store import SHA256_HEX as _SHA256_HEX
from fx1.serve.receipt_store import ReceiptIndex as _ReceiptIndex
from fx1.serve.webhooks import (
    WEBHOOK_SIGNATURE_HEADER,
    WEBHOOK_TIMESTAMP_HEADER,
    sign_webhook,
)
from quant_fund.research.receipt_v2 import verify_receipt_file, verify_receipt_payload

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
_RECEIPTS_DIR_ENV = "FX1_API_RECEIPTS_DIR"
_BYOK_OVERRIDE_ENV = "FX1_API_BYOK_OVERRIDE"

# Headers browser clients can read off responses when CORS is enabled.
_CORS_EXPOSE_HEADERS = [
    "ETag",
    "Location",
    "Retry-After",
    "X-Fx1-Api-Version",
    "X-Fx1-Completion-Id",
    "X-Fx1-Receipt-Valid",
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


class BackendAttempt(_Model):
    """One link of a backend fallback chain: which name was tried and how
    it ended (``error_class`` carries the verdict on a failed link)."""

    backend: str
    ok: bool
    error_class: str | None = None
    latency_ms: float | None = None


def _fallback_chain_valid(
    backend: str,
    fallbacks: Sequence[str],
    checkpoint_dir: str | None,
    byok: ByokOverride | None,
) -> None:
    """Chain validation shared by the sync and batch request models."""
    if len(set(fallbacks)) != len(fallbacks) or backend in fallbacks:
        raise ValueError("fallbacks must be distinct and not repeat the primary backend")
    chain = {backend, *fallbacks}
    if byok is not None and "byok" not in chain:
        raise ValueError("a byok override applies only to a 'byok' link in the chain")
    if checkpoint_dir is not None and "local_fx1" not in chain:
        raise ValueError("checkpoint_dir applies only to a 'local_fx1' link in the chain")


def _sampling_of(body: CompleteRequest | CompleteBatchRequest) -> SamplingParams:
    """Decode params declared on the request → the dataclass the backends
    take. The wire-facing flat fields are validated by pydantic; the
    resolved set is what the completion record seals."""
    return SamplingParams(
        temperature=body.temperature,
        top_p=body.top_p,
        max_tokens=body.max_tokens,
        seed=body.seed,
    )


class CompleteRequest(_Model):
    backend: Literal["hosted_k3", "local_fx1", "byok"]
    messages: list[ChatMessage] = Field(min_length=1, max_length=512)
    checkpoint_dir: str | None = None
    receipt_hashes: list[str] | None = None
    # Per-call credentials; only meaningful with backend="byok".
    byok: ByokOverride | None = None
    # Per-call backend deadline; beats each backend's env/config default.
    timeout_s: float | None = Field(default=None, gt=0, le=3600)
    # Ordered alternates tried after the primary — only on availability
    # faults (unconfigured / transport / circuit open). A gate refusal or
    # a client error aborts the request; a refusal is a verdict, not a
    # reason to spend another backend's capacity.
    fallbacks: list[Literal["hosted_k3", "local_fx1", "byok"]] = Field(
        default_factory=list, max_length=2
    )
    # Declared decode params — only fields set here reach the wire beyond
    # temperature (a provider that doesn't know ``seed`` never sees it);
    # the resolved set lands in the completion record as evidence of what
    # was sampled. Unset temperature defaults to 0 (eval runs stay
    # deterministic unless the caller opts out).
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0, le=262144)
    seed: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _chain_valid(self) -> CompleteRequest:
        _fallback_chain_valid(self.backend, self.fallbacks, self.checkpoint_dir, self.byok)
        return self


class CompleteResponse(_Model):
    backend: str
    model: str | None
    content: str
    receipt_hashes: list[str]
    # wall-clock ms inside the backend call — fx-1 sees per-call cost without
    # timing middleware overhead; idempotent replays report the original call's
    # latency alongside ``replayed``.
    latency_ms: float
    # Token counts reported by the endpoint for THIS call (prompt_tokens /
    # completion_tokens / total_tokens where the provider supplies them);
    # None when the backend has no usage channel — never fabricated.
    usage: dict[str, int] | None = None
    # True when the response came from the Idempotency-Key cache — lets
    # fx-1 audit retried calls without paying for them twice.
    replayed: bool = False
    # Server-minted handle into the completion log
    # (``GET /harness/completions/{id}``) — replays keep the original id.
    completion_id: str | None = None
    # Ordered chain trace — every link tried (last is the serving link),
    # empty when the primary served unchallenged.
    attempts: list[BackendAttempt] = []
    # The resolved decode params actually sent to the provider.
    sampling: dict[str, Any] | None = None


class CompleteBatchRequest(_Model):
    backend: Literal["hosted_k3", "local_fx1", "byok"]
    batch: list[list[ChatMessage]] = Field(min_length=1, max_length=64)
    checkpoint_dir: str | None = None
    receipt_hashes: list[str] | None = None
    byok: ByokOverride | None = None
    timeout_s: float | None = Field(default=None, gt=0, le=3600)
    max_workers: int = Field(default=4, ge=1, le=16)
    # Resolve-level chain: first resolvable link serves the whole batch
    # (one backend per batch — per-item failover can't attribute usage).
    fallbacks: list[Literal["hosted_k3", "local_fx1", "byok"]] = Field(
        default_factory=list, max_length=2
    )

    # Declared decode params applied to every item — same semantics as
    # ``CompleteRequest``; the resolved set is echoed on the response and
    # on each item's completion record.
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0, le=262144)
    seed: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _chain_valid(self) -> CompleteBatchRequest:
        _fallback_chain_valid(self.backend, self.fallbacks, self.checkpoint_dir, self.byok)
        return self


class BackendProbeRequest(_Model):
    """Optional controls for a backend liveness probe — same credential
    plumbing as a completion, so a BYOK probe tests the caller's real
    endpoint. ``prompt`` defaults to a one-token ping."""

    checkpoint_dir: str | None = None
    byok: ByokOverride | None = None
    timeout_s: float | None = Field(default=None, gt=0, le=3600)
    prompt: str = Field(default="ping", min_length=1, max_length=256)


class BackendProbeResponse(_Model):
    """Deep-health verdict for one backend — whether a real minimal
    completion succeeded, how long it took, and why not when it didn't."""

    backend: str
    ok: bool
    model: str | None = None
    latency_ms: float
    error: str | None = None
    error_class: str | None = None


class GateCheckRequest(_Model):
    """Text to run through the honesty gate — pre-flight for writers
    before they spend model tokens (or for validators on the way out)."""

    text: str = Field(min_length=0, max_length=262144)


class GateCheckResponse(_Model):
    """Gate verdict — ``ok`` mirrors whether the text would pass the gate;
    ``error`` carries the refusal reason when it wouldn't."""

    ok: bool
    error: str | None = None


class EvalSubmitRequest(_Model):
    """Async eval submission: one seeded suite against a backend chain.

    The suite runs on the shared jobs executor under the same drain/cap
    contract as run jobs; poll ``GET /harness/evals/{eval_id}`` and export
    the sealed ``fx1_eval_record.v1`` doc at ``/receipt`` once terminal.
    Evals always run under the decode pin ``{"temperature": 0.0}`` — eval
    evidence is deterministic evidence; the pin is recorded on the
    record.
    """

    suite: EvalSuiteName
    backend: Literal["hosted_k3", "local_fx1", "byok"]
    seed: int = Field(default=0, ge=0)
    checkpoint_dir: str | None = None
    byok: ByokOverride | None = None
    timeout_s: float | None = Field(default=None, gt=0, le=3600)
    fallbacks: list[Literal["hosted_k3", "local_fx1", "byok"]] = Field(
        default_factory=list, max_length=2
    )
    # Optional grader link — only the judge suites (capability, ext_bench)
    # consume it. Resolved once, unchained; ``judge_byok`` binds only a
    # 'byok' judge.
    judge_backend: Literal["hosted_k3", "local_fx1", "byok"] | None = None
    judge_byok: ByokOverride | None = None
    # Terminal-state webhook (the job contract): the finished record is
    # POSTed to ``callback_url`` on every terminal transition;
    # ``callback_secret`` HMAC-signs the delivery and is never echoed.
    callback_url: str | None = None
    callback_secret: str | None = None

    @field_validator("callback_url")
    @classmethod
    def _eval_callback_url_http(cls, v: str | None) -> str | None:
        if v is None:
            return v
        parsed = urllib.parse.urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"callback_url must be an http(s) URL with a host, got {v!r}")
        return v

    @model_validator(mode="after")
    def _eval_valid(self) -> EvalSubmitRequest:
        _fallback_chain_valid(self.backend, self.fallbacks, self.checkpoint_dir, self.byok)
        if self.judge_byok is not None and self.judge_backend != "byok":
            raise ValueError("judge_byok applies only to judge_backend='byok'")
        if self.judge_backend is not None and not suite_accepts_judge(self.suite):
            raise ValueError(f"suite '{self.suite}' takes no judge")
        if self.callback_secret is not None and not self.callback_url:
            raise ValueError("callback_secret requires callback_url")
        return self


class EvalSubmitResponse(_Model):
    """Submission ack — ``replayed`` marks an Idempotency-Key hit (the
    eval ran once already; no second execution)."""

    eval_id: str
    status: str
    replayed: bool = False


class EvalListResponse(_Model):
    """Eval inventory page: ``total`` is the filtered count before paging."""

    records: list[EvalRecord]
    total: int


class CompleteBatchItem(_Model):
    ok: bool
    latency_ms: float
    content: str | None = None
    error: str | None = None
    error_class: str | None = None
    completion_id: str | None = None


class CompleteBatchResponse(_Model):
    backend: str
    model: str | None
    receipt_hashes: list[str]
    results: list[CompleteBatchItem]
    # The resolved decode params sent for every item in the batch.
    sampling: dict[str, Any] | None = None
    # Sum of the backend's reported usage across this batch (a shared
    # endpoint can't attribute counts per item under worker threads —
    # only the batch-level delta is honest). None when the backend is
    # silent on usage.
    usage_total: dict[str, int] | None = None
    replayed: bool = False
    # Resolve-level chain trace — the link that served the batch.
    attempts: list[BackendAttempt] = []


class CompletionRecord(_Model):
    """One recorded model call: hashes of what went in and came out,
    latency, reported usage, and the verdict — the harness's own calls
    are auditable evidence. Content itself is never stored (hashes only),
    and the log is a bounded in-process ring."""

    completion_id: str
    backend: str
    model: str | None = None
    ok: bool
    latency_ms: float
    at: float
    usage: dict[str, int] | None = None
    error: str | None = None
    error_class: str | None = None
    prompt_sha256: str
    output_sha256: str | None = None
    # Chain trace when a fallback chain ran — sealed evidence of failover.
    attempts: list[BackendAttempt] | None = None
    # The resolved decode params sent to the provider.
    sampling: dict[str, Any] | None = None


class CompletionListResponse(_Model):
    items: list[CompletionRecord]
    count: int


class ReceiptIndexItem(_Model):
    sha256: str
    name: str


class ReceiptIndexResponse(_Model):
    items: list[ReceiptIndexItem]
    count: int


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


class BackendCompletionStats(_Model):
    """Per-backend completion accounting: outcome counts, a cumulative
    latency histogram (edges are ``_LAT_BUCKETS_MS`` plus ``+Inf``), and
    backend-reported token sums. ``usage_calls`` counts calls that carried
    a usage dict — a silent provider shows zero tokens AND zero calls."""

    ok: int = 0
    error: int = 0
    latency_count: int = 0
    latency_sum_ms: float = 0.0
    latency_buckets: dict[str, int] = {}
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    usage_calls: int = 0


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
    # completion calls only — per backend name as requested on the wire
    complete: dict[str, BackendCompletionStats] = {}


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
    eval_suites: list[str]
    limits: dict[str, float]
    backends: dict[str, bool]
    roles: list[str]


class BackendProbeVerdict(_Model):
    """The most recent ``/harness/backends/{name}/probe`` outcome — kept
    process-local so a monitoring scrape can read the last deep-health
    verdict without spending another live call."""

    ok: bool
    latency_ms: float
    checked_at: float
    error_class: str | None = None


class BackendStatusEntry(_Model):
    """One backend's liveness surface: whether it is configured and, when
    the circuit breaker is enabled, whether it is currently fast-failing."""

    configured: bool
    circuit_open: bool
    cooldown_remaining_s: float
    consecutive_failures: int
    last_probe: BackendProbeVerdict | None = None


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


_LAT_BUCKETS_MS = (100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 30000.0, 60000.0)


class _Metrics:
    """Request counters + inflight gauge + per-backend completion
    histograms, shared via app.state."""

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
        # backend -> {"ok","error","count","sum","buckets" (cumulative),
        #              "*_tokens","usage_calls"}
        self._complete: dict[str, dict[str, Any]] = {}
        self.draining = threading.Event()

    def record_complete(
        self,
        backend: str,
        ok: bool,
        latency_ms: float,
        usage: dict[str, int] | None = None,
    ) -> None:
        """One attempted model call: outcome + latency into the backend's
        cumulative histogram, plus backend-reported token sums when the
        endpoint supplies them. Called only when the backend was invoked —
        breaker rejections and pre-call validation aren't model work.
        ``usage_calls`` counts calls that reported usage at all, so a
        silent provider is distinguishable from a zero bill."""
        with self._cond:
            st = self._complete.setdefault(
                backend,
                {
                    "ok": 0,
                    "error": 0,
                    "count": 0,
                    "sum": 0.0,
                    # one entry per edge + a final +Inf slot
                    "buckets": [0] * (len(_LAT_BUCKETS_MS) + 1),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "usage_calls": 0,
                },
            )
            st["ok" if ok else "error"] += 1
            st["count"] += 1
            st["sum"] += latency_ms
            for i, edge in enumerate(_LAT_BUCKETS_MS):
                if latency_ms <= edge:
                    st["buckets"][i] += 1
            st["buckets"][-1] += 1
            if usage:
                st["usage_calls"] += 1
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    v = usage.get(k)
                    if isinstance(v, int) and not isinstance(v, bool):
                        st[k] += v

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
                complete={
                    b: BackendCompletionStats(
                        ok=st["ok"],
                        error=st["error"],
                        latency_count=st["count"],
                        latency_sum_ms=round(st["sum"], 3),
                        latency_buckets={
                            **_lat_bucket_labels(st["buckets"]),
                        },
                        prompt_tokens=st["prompt_tokens"],
                        completion_tokens=st["completion_tokens"],
                        total_tokens=st["total_tokens"],
                        usage_calls=st["usage_calls"],
                    )
                    for b, st in sorted(self._complete.items())
                },
            )


def _lat_bucket_labels(counts: list[int]) -> dict[str, int]:
    """Cumulative bucket counts keyed by le-label, ``+Inf`` last — the
    Prometheus histogram convention, kept identical on the JSON snapshot."""
    labels = [f"{edge:g}" for edge in _LAT_BUCKETS_MS] + ["+Inf"]
    return dict(zip(labels, counts, strict=True))


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


def _body_fp(body: BaseModel, *, exclude: set[str] | None = None) -> str:
    """Fingerprint for the idempotency contract — hashed so a stored
    dedupe record can never carry a secret (per-request BYOK keys)."""
    return hashlib.sha256(body.model_dump_json(exclude=exclude).encode()).hexdigest()


def _openai_sse(
    body: OpenAIChatRequest,
    *,
    content: str,
    backend: str,
    model: str | None,
    usage: dict[str, int] | None,
    cid: str,
    created: int | None = None,
) -> Iterator[str]:
    """Serialize ``openai_chunks`` payloads into SSE ``data:`` frames +
    the terminal ``[DONE]`` marker. The chunk payloads themselves are
    generated by the shared compat layer (``openai_chunks``), so the wire
    and the SDK emit the same sequence. ``created`` pins the chunk
    timestamp — idempotent replays pass the stored envelope's value so a
    resumed stream is byte-identical."""

    def _frame(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

    for payload in openai_chunks(
        text=content,
        backend=backend,
        model=model,
        cid=cid,
        include_usage=bool((body.stream_options or {}).get("include_usage")),
        usage=usage,
        created=created,
    ):
        yield _frame(payload)
    yield "data: [DONE]\n\n"


def _breaker_key_name(name: str, byok: ByokOverride | None) -> str:
    """Backend key for one chain link — a BYOK endpoint gets its own
    circuit keyed by endpoint hash wherever it sits in the chain."""
    if byok is None or name != "byok":
        return name
    return f"{name}:{hashlib.sha256(byok.base_url.encode()).hexdigest()[:16]}"


def _breaker_key(body: CompleteRequest | CompleteBatchRequest) -> str:
    """Backend key for the circuit breaker. A per-request BYOK override gets
    its own circuit keyed by endpoint hash — one caller's dead endpoint must
    never fast-fail another caller's BYOK or the env-configured default."""
    return _breaker_key_name(body.backend, body.byok)


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


def _deliver_callback(rec: JobStatusResponse | EvalRecord) -> None:
    """Terminal-state webhook: POST the full record to the caller's
    ``callback_url`` — the shared contract for jobs and evals.
    Best-effort — a dead or slow endpoint records
    ``callback_status='failed'`` on the record, never raises into the
    worker and never changes the record's own status. Transient faults
    (network errors, 5xx) retry ``_WEBHOOK_MAX_ATTEMPTS`` times with
    capped backoff; a 4xx is a definitive rejection and is never
    retried."""
    url = rec.callback_url
    if not url:
        return
    for attempt in range(_WEBHOOK_MAX_ATTEMPTS):
        if attempt:
            time.sleep(_WEBHOOK_BACKOFF_S * (1 << (attempt - 1)))
        rec.callback_attempts = attempt + 1
        try:
            payload = rec.model_dump_json().encode()
            headers = {"Content-Type": "application/json"}
            if rec._callback_secret:
                ts = str(int(time.time()))
                headers[WEBHOOK_TIMESTAMP_HEADER] = ts
                headers[WEBHOOK_SIGNATURE_HEADER] = sign_webhook(rec._callback_secret, ts, payload)
            req = urllib.request.Request(
                url,
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310  # nosec B310 — caller-declared webhook target, validated http(s) at submit
                if resp.status < 400:
                    rec.callback_status = "delivered"
                    rec.callback_error = None
                    return
                rec.callback_status = "failed"
                rec.callback_error = f"callback endpoint returned {resp.status}"
                if 400 <= resp.status < 500:
                    return  # definitive rejection — never retried
        except Exception as exc:  # noqa: BLE001 — delivery faults land on the record, not the worker
            rec.callback_status = "failed"
            rec.callback_error = f"{type(exc).__name__}: {exc}"


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
    body_fp = _body_fp(body, exclude={"idempotency_key"})
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
            _deliver_callback(job)
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
    eval_store: EvalStore,
    jobs_executor: ThreadPoolExecutor,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Graceful-exit contract: on shutdown the gate drains (new work gets
    503), every still-queued job flips to 'cancelled' and fires its
    signed webhook, queued evals flip to 'cancelled', and the executor
    releases pending futures. Running work isn't interrupted — it
    finishes bounded by its command timeout or dies with the process."""

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        metrics.draining.set()
        for pending in job_store.cancel_pending():
            _deliver_callback(pending)
        for pending_eval in eval_store.cancel_pending():
            _deliver_callback(pending_eval)
        jobs_executor.shutdown(wait=False, cancel_futures=True)

    return _lifespan


def _mount_receipt_routes(app: FastAPI, receipt_index: _ReceiptIndex) -> None:
    """Receipt-verify + content-addressed fetch routes, kept out of
    ``create_app`` to keep its branch complexity under the repo's ruff cap."""

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

    @app.get(
        "/receipts",
        response_model=ReceiptIndexResponse,
        tags=["receipts"],
        operation_id="receipts_index",
    )
    def receipts_index() -> ReceiptIndexResponse:
        """List the store's sealed receipts: sha256 → filename, sorted."""
        if not receipt_index.available():
            raise ApiError(503, "receipts store unavailable", code="receipts_unavailable")
        items = receipt_index.items()
        return ReceiptIndexResponse(
            items=[ReceiptIndexItem(sha256=sha, name=path.name) for sha, path in items],
            count=len(items),
        )

    @app.get(
        "/receipts/{sha256}",
        tags=["receipts"],
        operation_id="receipt_fetch",
        response_class=Response,
    )
    def receipt_fetch(sha256: str, request: Request) -> Response:
        """Serve the sealed receipt addressed by its content hash.

        The body is the receipt's committed bytes verbatim, and receipts are
        content-addressed by their seal — so ``ETag`` is the hash itself and
        the response is ``Cache-Control: immutable``. Validity under the live
        verifier is reported on ``X-Fx1-Receipt-Valid`` so a caller learns
        the receipt it fetched still verifies without a second round trip.
        """
        if not _SHA256_HEX.fullmatch(sha256):
            raise ApiError(422, "sha256 must be 64 lowercase hex", code="invalid_sha256")
        if not receipt_index.available():
            raise ApiError(503, "receipts store unavailable", code="receipts_unavailable")
        path = receipt_index.lookup(sha256)
        if path is None:
            raise ApiError(404, "receipt not found", code="receipt_not_found")
        inm = request.headers.get("if-none-match", "")
        if inm.strip() == "*" or f'"{sha256}"' in inm:
            return Response(
                status_code=304,
                headers={"ETag": f'"{sha256}"', "Cache-Control": "public, immutable"},
            )
        try:
            body = path.read_bytes()
        except OSError as exc:
            raise ApiError(503, "receipts store read failed", code="receipts_unavailable") from exc
        valid = bool(verify_receipt_file(path)["valid"])
        return Response(
            content=body,
            media_type="application/json",
            headers={
                "ETag": f'"{sha256}"',
                "Cache-Control": "public, immutable",
                "X-Fx1-Receipt-Valid": "true" if valid else "false",
            },
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

    @app.get(
        "/harness/jobs/{job_id}/receipt",
        tags=["jobs"],
        operation_id="job_receipt",
    )
    def job_receipt(job_id: str) -> dict[str, Any]:
        """Export the job's ledger record as a sealed
        ``fx1_job_record.v1`` document — the terminal ``result`` embeds
        with its streams digested (stdout/stderr sha256, never content).
        Verify with ``POST /receipts/verify`` or the SDK."""
        from fx1.serve.ops_receipt import job_record_receipt  # noqa: PLC0415

        job = job_store.get(job_id)
        if job is None:
            raise ApiError(404, f"unknown job_id {job_id!r}")
        return job_record_receipt(job.model_dump(mode="json"))

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
        _deliver_callback(job)  # cancelled is terminal — fire the webhook
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


class _OpenAIIdemRecord(_Model):
    """Stored ``/v1/chat/completions`` outcome for ``Idempotency-Key``
    replay. The minted envelope is the source of truth: JSON responses
    replay it verbatim and SSE chunk payloads regenerate
    deterministically from it (same ``id``/``created``/content — a
    replayed stream is byte-identical to the first send). Only
    successful completions are stored: a gate refusal or backend fault
    is never pinned, so retrying a failed call re-executes."""

    envelope: dict[str, Any]
    # ``_idem_lookup`` stamps this on a cache hit, matching the other
    # idempotent routes' ``replayed`` convention (surfaced on the wire
    # as the ``X-Fx1-Idempotent-Replay`` header).
    replayed: bool = False


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
    # no-store is the default posture; a route that deliberately declares a
    # caching policy (immutable content-addressed bytes) wins over it.
    if "cache-control" not in response.headers:
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


_COMPLETION_LOG_MAX = 256


class _CompletionLog:
    """Bounded in-process ring of per-call completion records. Hashes of
    prompt/output are stored, never the content — the log is evidence,
    not a transcript. Probes stay out of it: they already carry their own
    surface (``probe:<name>`` metrics + the status-cache verdicts)."""

    def __init__(self, cap: int = _COMPLETION_LOG_MAX) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._items: dict[str, CompletionRecord] = {}

    def append(self, rec: CompletionRecord) -> None:
        with self._lock:
            self._items[rec.completion_id] = rec
            while len(self._items) > self._cap:
                self._items.pop(next(iter(self._items)))

    def get(self, completion_id: str) -> CompletionRecord | None:
        with self._lock:
            return self._items.get(completion_id)

    def latest(self, limit: int, backend: str | None) -> list[CompletionRecord]:
        with self._lock:
            items = sorted(self._items.values(), key=lambda r: r.at, reverse=True)
        if backend is not None:
            items = [r for r in items if r.backend == backend]
        return items[:limit]


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
    lines += [
        "# HELP fx1_complete_total Model calls attempted, by backend and outcome.",
        "# TYPE fx1_complete_total counter",
        "# HELP fx1_complete_latency_ms Model-call latency histogram (ms), per backend.",
        "# TYPE fx1_complete_latency_ms histogram",
    ]
    for b in sorted(snap.complete):
        c = snap.complete[b]
        lb = _label(b)
        lines.append(f'fx1_complete_total{{backend="{lb}",outcome="ok"}} {c.ok}')
        lines.append(f'fx1_complete_total{{backend="{lb}",outcome="error"}} {c.error}')
        for le, n in c.latency_buckets.items():
            lines.append(f'fx1_complete_latency_ms_bucket{{backend="{lb}",le="{le}"}} {n}')
        lines.append(f'fx1_complete_latency_ms_sum{{backend="{lb}"}} {c.latency_sum_ms}')
        lines.append(f'fx1_complete_latency_ms_count{{backend="{lb}"}} {c.latency_count}')
    lines += [
        "# HELP fx1_complete_tokens_total Backend-reported token sums, per backend and kind.",
        "# TYPE fx1_complete_tokens_total counter",
        "# HELP fx1_complete_usage_calls_total Model calls that carried a usage report.",
        "# TYPE fx1_complete_usage_calls_total counter",
    ]
    for b in sorted(snap.complete):
        c = snap.complete[b]
        lb = _label(b)
        lines.append(f'fx1_complete_tokens_total{{backend="{lb}",kind="prompt"}} {c.prompt_tokens}')
        lines.append(
            f'fx1_complete_tokens_total{{backend="{lb}",kind="completion"}} {c.completion_tokens}'
        )
        lines.append(f'fx1_complete_tokens_total{{backend="{lb}",kind="total"}} {c.total_tokens}')
        lines.append(f'fx1_complete_usage_calls_total{{backend="{lb}"}} {c.usage_calls}')
    return "\n".join(lines) + "\n"


def _close_backend(backend: Any) -> None:
    closer = getattr(backend, "close", None)
    if callable(closer):
        closer()


def _mount_complete_routes(  # noqa: C901 — eval submission shares the chain/job surface
    app: FastAPI,
    *,
    slot: Callable[[], Iterator[None]],
    resolve_backend: Callable[[str, str | None, dict[str, str] | None, float | None], Any],
    sse_keepalive_s: float,
    complete_idem_store: _IdemStore[CompleteResponse],
    complete_batch_idem_store: _IdemStore[CompleteBatchResponse],
    openai_idem_store: _IdemStore[_OpenAIIdemRecord],
    breaker: _BackendBreaker | None,
    receipt_index: _ReceiptIndex,
    metrics: _Metrics,
    probe_cache: dict[str, BackendProbeVerdict],
    probe_lock: threading.Lock,
    completion_log: _CompletionLog,
    eval_store: EvalStore,
    inflight: threading.BoundedSemaphore,
    jobs_executor: ThreadPoolExecutor,
) -> None:
    """Complete routes (sync / SSE stream / batch) + eval submissions —
    extracted from ``create_app`` to keep its branch complexity under the
    ruff cap. Evals share the complete chain resolution and the jobs
    executor's slot contract."""

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

    def _check_citations(hashes: list[str] | None) -> None:
        """Cited evidence must resolve against the mounted receipt store:
        a completion may not footnote a receipt the server cannot produce.
        Runs before the breaker and the backend call — a bogus citation is a
        422 client fault and never burns a slot or trips the circuit. When no
        store is mounted, citations stay advisory (nothing to check against)."""
        if not hashes or not receipt_index.available():
            return
        missing = [
            h for h in hashes if not _SHA256_HEX.fullmatch(h) or receipt_index.lookup(h) is None
        ]
        if missing:
            raise ApiError(
                422,
                f"unresolvable receipt citations: {missing}",
                code="receipt_not_found",
            )

    def _resolve_candidate(
        name: str, body: CompleteRequest | CompleteBatchRequest | EvalSubmitRequest
    ) -> Any:
        """Resolve one chain link — per-link kwargs: the byok override binds
        only a 'byok' link, checkpoint_dir only a 'local_fx1' link."""
        return resolve_backend(
            name,
            body.checkpoint_dir if name == "local_fx1" else None,
            body.byok.model_dump() if name == "byok" and body.byok is not None else None,
            body.timeout_s,
        )

    def _resolve_chain(
        body: CompleteRequest | CompleteBatchRequest | EvalSubmitRequest,
    ) -> tuple[str, Any, list[BackendAttempt]]:
        """First chain link that admits + resolves serves; a 503
        (unconfigured / unavailable / circuit open) records the attempt and
        moves on. Any other error is a request fault and aborts."""
        attempts: list[BackendAttempt] = []
        last: ApiError | None = None
        for cand in [body.backend, *body.fallbacks]:
            key = _breaker_key_name(cand, body.byok)
            try:
                _breaker_admit(key)
                backend = _resolve_candidate(cand, body)
            except ApiError as exc:
                if exc.status_code == 503:
                    attempts.append(
                        BackendAttempt(backend=cand, ok=False, error_class="backend_unavailable")
                    )
                    if breaker is not None:
                        breaker.report(key, False)
                    last = exc
                    continue
                raise
            attempts.append(BackendAttempt(backend=cand, ok=True))
            return cand, backend, attempts
        assert last is not None  # noqa: S101 — every link failed retriably
        raise last

    def _submit_eval(
        body: EvalSubmitRequest,
        idempotency_key: str | None,
    ) -> EvalSubmitResponse:
        """Eval submission core — the job contract (idempotency lookup ->
        drain check -> slot admission -> background execution) applied to
        the eval suites. The slot is held for the eval's lifetime and
        released by the worker, so evals queue no deeper than
        ``max_inflight``."""
        key = (idempotency_key or "").strip() or None
        if key is not None and len(key) > _IDEM_KEY_MAX:
            raise ApiError(400, "Idempotency-Key must be <= 256 chars")
        body_fp = _body_fp(body)
        if key is not None:
            entry = eval_store.get_key(key)
            if entry is not None:
                fp, eval_id = entry
                if fp != body_fp:
                    raise ApiError(
                        409,
                        "Idempotency-Key reuse with a different request body",
                    )
                rec = eval_store.get(eval_id)
                if rec is not None:
                    return EvalSubmitResponse(eval_id=eval_id, status=rec.status, replayed=True)
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
        record = EvalRecord(
            eval_id=uuid.uuid4().hex,
            suite=body.suite,
            backend=body.backend,
            seed=body.seed,
            status="queued",
            created_at=time.time(),
            sampling=EVAL_SAMPLING.body_fields(),
            callback_url=body.callback_url,
        )
        record._callback_secret = body.callback_secret

        def _exec() -> None:
            if record.status == "cancelled":
                metrics.release()
                inflight.release()
                return
            record.status = "running"
            try:
                name, backend, attempts = _resolve_chain(body)
                record.backend = name
                record.attempts = [a.model_dump(mode="json") for a in attempts]
                model = metered_model(
                    backend,
                    metric_key=f"eval:{body.suite}:{name}",
                    record=metrics.record_complete,
                )
                judge = None
                if body.judge_backend is not None:
                    judge_key = _breaker_key_name(body.judge_backend, body.judge_byok)
                    _breaker_admit(judge_key)
                    judge_obj = resolve_backend(
                        body.judge_backend,
                        None,
                        (
                            body.judge_byok.model_dump()
                            if body.judge_backend == "byok" and body.judge_byok is not None
                            else None
                        ),
                        body.timeout_s,
                    )
                    judge = metered_model(
                        judge_obj,
                        metric_key=f"eval:{body.suite}:judge:{body.judge_backend}",
                        record=metrics.record_complete,
                    )
                run_eval_record(record, model=model, judge=judge)
            except ApiError as exc:
                record.error = f"{exc.status_code}: {exc.detail}"
                record.status = "failed"
                record.finished_at = time.time()
            except Exception as exc:  # noqa: BLE001 — worker faults land in the record
                record.error = f"{type(exc).__name__}: {exc}"
                record.status = "failed"
                record.finished_at = time.time()
            finally:
                _deliver_callback(record)
            metrics.release()
            inflight.release()

        try:
            jobs_executor.submit(_exec)
        except RuntimeError as exc:  # executor gone (shutdown race)
            metrics.release()
            inflight.release()
            raise ApiError(503, "job executor unavailable", code="over_capacity") from exc
        eval_store.put(record, key, body_fp)
        return EvalSubmitResponse(eval_id=record.eval_id, status=record.status, replayed=False)

    @app.post(
        "/harness/evals",
        response_model=EvalSubmitResponse,
        status_code=202,
        tags=["evals"],
        operation_id="submit_eval",
    )
    def submit_eval(
        body: EvalSubmitRequest,
        response: Response,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> EvalSubmitResponse:
        """Submit an eval suite against a backend chain — same
        drain/cap/Idempotency-Key contract as job submission. The record
        carries the suite, seed, serving backend, chain attempts, decode
        pin, and the serialized report once terminal; export it sealed at
        ``GET /harness/evals/{eval_id}/receipt``."""
        out = _submit_eval(body, idempotency_key)
        response.headers["Location"] = f"/harness/evals/{out.eval_id}"
        return out

    @app.get(
        "/harness/evals",
        response_model=EvalListResponse,
        tags=["evals"],
        operation_id="list_evals",
    )
    def list_evals(
        status: Literal["queued", "running", "succeeded", "failed", "cancelled"] | None = None,
        suite: EvalSuiteName | None = None,
        limit: int = Query(default=100, ge=1, le=256),
    ) -> EvalListResponse:
        """Newest-first eval inventory, filterable by status and suite."""
        records, total = eval_store.list_records(status=status, suite=suite, limit=limit)
        return EvalListResponse(records=records, total=total)

    @app.get(
        "/harness/evals/{eval_id}",
        response_model=EvalRecord,
        tags=["evals"],
        operation_id="get_eval",
    )
    def get_eval(eval_id: str) -> EvalRecord:
        rec = eval_store.get(eval_id)
        if rec is None:
            raise ApiError(404, f"unknown eval_id {eval_id!r}")
        return rec

    @app.get(
        "/harness/evals/{eval_id}/receipt",
        tags=["evals"],
        operation_id="eval_receipt",
    )
    def eval_receipt(eval_id: str) -> dict[str, Any]:
        """Export the eval record as a sealed ``fx1_eval_record.v1``
        document — terminal records only: a still-running eval's receipt
        would seal a mutable report. Verify with ``POST /receipts/verify``
        or the SDK."""
        rec = eval_store.get(eval_id)
        if rec is None:
            raise ApiError(404, f"unknown eval_id {eval_id!r}")
        if rec.status not in _TERMINAL_JOB_STATUS:
            raise ApiError(
                409,
                f"eval {eval_id!r} is {rec.status} — receipts export on terminal records only",
                code="eval_not_terminal",
            )
        return eval_record_receipt(rec.model_dump(mode="json"))

    @app.delete(
        "/harness/evals/{eval_id}",
        response_model=EvalRecord,
        tags=["evals"],
        operation_id="cancel_eval",
    )
    def cancel_eval(eval_id: str) -> EvalRecord:
        """Cooperative cancel: a queued eval flips to 'cancelled' and its
        executor slot frees on dequeue. Running and terminal evals 409 —
        suite runners have no mid-run kill handle."""
        rec, outcome = eval_store.cancel(eval_id)
        if rec is None:
            raise ApiError(404, f"unknown eval_id {eval_id!r}")
        if outcome != "cancelled":
            raise ApiError(409, f"eval {eval_id!r} is {outcome}")
        _deliver_callback(rec)  # cancelled is terminal — fire the webhook
        return rec

    @app.post(
        "/harness/complete",
        response_model=CompleteResponse,
        tags=["complete"],
        operation_id="complete",
    )
    def complete(
        body: CompleteRequest,
        response: Response,
        _slot_held: None = Depends(slot),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> CompleteResponse:
        # Same Idempotency-Key contract as the runs route: retried
        # completions replay from the cache instead of re-billing the model.
        body_fp = _body_fp(body)
        key, replay = _idem_lookup(idempotency_key, complete_idem_store, body_fp)
        if replay is not None:
            if replay.completion_id is not None:
                response.headers["X-Fx1-Completion-Id"] = replay.completion_id
            return replay
        _check_citations(body.receipt_hashes)
        messages = [{"role": m.role, "content": m.content} for m in body.messages]
        cid = uuid.uuid4().hex
        prompt_sha256 = hashlib.sha256(
            json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        rec_err: str | None = None
        rec_cls: str | None = None
        content = ""
        serving: str | None = None
        usage_snap: Any = None
        model_snap: Any = None
        call_latency_ms = 0.0
        attempts: list[BackendAttempt] = []
        last_exc: ApiError | None = None
        sampling_params = _sampling_of(body)
        sampling_fields = sampling_params.body_fields()
        # Ordered fallback chain: each link gets its own admit + resolve +
        # call. Only availability faults advance the chain — a gate
        # refusal, a capability gap (501), or a client error aborts.
        try:
            for cand in [body.backend, *body.fallbacks]:
                cand_key = _breaker_key_name(cand, body.byok)
                backend: Any = None
                try:
                    _breaker_admit(cand_key)
                    backend = _resolve_candidate(cand, body)
                except ApiError as exc:
                    # only backend-unavailable counts — client errors (404
                    # unknown backend, 422 bad args) must never trip the
                    # circuit, or a caller could deny the backend for
                    # everyone by spamming bad requests.
                    if exc.status_code == 503:
                        rec_cls = "backend_unavailable"
                        rec_err = str(exc)
                        attempts.append(
                            BackendAttempt(
                                backend=cand, ok=False, error_class="backend_unavailable"
                            )
                        )
                        if breaker is not None:
                            breaker.report(cand_key, False)
                        last_exc = exc
                        continue
                    raise
                t0 = time.monotonic()
                try:
                    content = cited_complete(
                        backend,
                        messages,
                        receipt_hashes=body.receipt_hashes,
                        sampling=sampling_params,
                    )
                except NotImplementedError as exc:
                    rec_cls = "not_supported"
                    rec_err = str(exc)
                    attempts.append(
                        BackendAttempt(backend=cand, ok=False, error_class="not_supported")
                    )
                    raise ApiError(501, str(exc)) from exc
                except Fx1HonestyError as exc:
                    rec_cls = "honesty_refusal"
                    rec_err = str(exc)
                    attempts.append(
                        BackendAttempt(backend=cand, ok=False, error_class="honesty_refusal")
                    )
                    # The model produced a contract-violating headline; the
                    # gate caught it before the bytes left — surface as 502,
                    # not success. Never falls back: a refusal is a verdict.
                    raise ApiError(
                        502,
                        f"honesty gate refused model output: {exc}",
                        code="honesty_gate",
                    ) from exc
                except (BackendNotConfiguredError, RuntimeError) as exc:
                    rec_cls = type(exc).__name__
                    rec_err = str(exc)
                    call_latency_ms = (time.monotonic() - t0) * 1000.0
                    attempts.append(
                        BackendAttempt(
                            backend=cand,
                            ok=False,
                            error_class=rec_cls,
                            latency_ms=call_latency_ms,
                        )
                    )
                    if breaker is not None:
                        breaker.report(cand_key, False)
                    last_exc = (
                        ApiError(503, str(exc), code="backend_unavailable")
                        if isinstance(exc, BackendNotConfiguredError)
                        else ApiError(502, str(exc), code="backend_failure")
                    )
                    continue
                finally:
                    _close_backend(backend)
                call_latency_ms = (time.monotonic() - t0) * 1000.0
                if breaker is not None:
                    breaker.report(cand_key, True)
                usage_snap = getattr(backend, "last_usage", None)
                model_snap = getattr(backend, "_model", None)
                serving = cand
                attempts.append(BackendAttempt(backend=cand, ok=True, latency_ms=call_latency_ms))
                break
            if serving is None:
                raise (
                    last_exc
                    if last_exc is not None
                    else ApiError(503, "no backend in the chain served")
                )
        finally:
            metrics.record_complete(
                serving or body.backend,
                serving is not None,
                call_latency_ms,
                usage=usage_snap if isinstance(usage_snap, dict) else None,
            )
            completion_log.append(
                CompletionRecord(
                    completion_id=cid,
                    backend=serving or body.backend,
                    model=model_snap if isinstance(model_snap, str) else None,
                    ok=serving is not None,
                    latency_ms=call_latency_ms,
                    at=time.time(),
                    usage=usage_snap if isinstance(usage_snap, dict) else None,
                    error=rec_err if serving is None else None,
                    error_class=rec_cls if serving is None else None,
                    prompt_sha256=prompt_sha256,
                    output_sha256=(
                        hashlib.sha256(content.encode("utf-8")).hexdigest()
                        if serving is not None
                        else None
                    ),
                    attempts=attempts if len(attempts) > 1 else None,
                    sampling=sampling_fields,
                )
            )
        assert serving is not None  # noqa: S101 — None already raised above
        resp = CompleteResponse(
            backend=serving,
            model=model_snap if isinstance(model_snap, str) else None,
            content=content,
            receipt_hashes=body.receipt_hashes or [],
            latency_ms=call_latency_ms,
            usage=usage_snap if isinstance(usage_snap, dict) else None,
            completion_id=cid,
            attempts=attempts if len(attempts) > 1 else [],
            sampling=sampling_fields,
        )
        response.headers["X-Fx1-Completion-Id"] = cid
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
        _check_citations(body.receipt_hashes)
        messages = [{"role": m.role, "content": m.content} for m in body.messages]
        sampling_params = _sampling_of(body)
        sampling_fields = sampling_params.body_fields()

        def _gather() -> tuple[list[str], str | None, float, dict[str, int] | None, str]:
            """Buffer + gate the backend stream; raises the mapped errors.

            The fallback chain applies at resolve level only — once a link
            is streaming there is no honest restart point."""
            serving, backend, chain_attempts = _resolve_chain(body)
            t0 = time.monotonic()
            call_ok = False
            cid = uuid.uuid4().hex
            prompt_sha256 = hashlib.sha256(
                json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            rec_err: str | None = None
            rec_cls: str | None = None
            try:
                if not isinstance(backend, StreamingBackend):
                    raise NotImplementedError(
                        f"backend {body.backend!r} does not support streaming"
                    )
                chunks = list(backend.stream(messages, sampling=sampling_params))
                joined = "".join(chunks)
                try:
                    validate_fx1_output(joined)
                except Fx1HonestyError as exc:
                    rec_cls = "honesty_refusal"
                    rec_err = str(exc)
                    raise ApiError(
                        502, f"honesty gate refused model output: {exc}", code="honesty_gate"
                    ) from exc
            except NotImplementedError as exc:
                rec_cls = "not_supported"
                rec_err = str(exc)
                raise ApiError(501, str(exc)) from exc
            except (BackendNotConfiguredError, RuntimeError) as exc:
                rec_cls = type(exc).__name__
                rec_err = str(exc)
                if breaker is not None:
                    breaker.report(_breaker_key_name(serving, body.byok), False)
                if isinstance(exc, BackendNotConfiguredError):
                    raise ApiError(503, str(exc), code="backend_unavailable") from exc
                raise ApiError(502, str(exc), code="backend_failure") from exc
            else:
                call_ok = True
                if breaker is not None:
                    breaker.report(_breaker_key_name(serving, body.byok), True)
            finally:
                usage_snap = getattr(backend, "last_usage", None)
                model_snap = getattr(backend, "_model", None)
                joined_snap = "".join(chunks) if call_ok else ""
                metrics.record_complete(
                    serving,
                    call_ok,
                    (time.monotonic() - t0) * 1000.0,
                    usage=usage_snap if isinstance(usage_snap, dict) else None,
                )
                completion_log.append(
                    CompletionRecord(
                        completion_id=cid,
                        backend=serving,
                        attempts=chain_attempts if len(chain_attempts) > 1 else None,
                        model=model_snap if isinstance(model_snap, str) else None,
                        ok=call_ok,
                        latency_ms=(time.monotonic() - t0) * 1000.0,
                        at=time.time(),
                        usage=usage_snap if isinstance(usage_snap, dict) else None,
                        error=rec_err,
                        error_class=rec_cls,
                        prompt_sha256=prompt_sha256,
                        sampling=sampling_fields,
                        output_sha256=(
                            hashlib.sha256(joined_snap.encode("utf-8")).hexdigest()
                            if call_ok
                            else None
                        ),
                    )
                )
                _close_backend(backend)
            if body.receipt_hashes:
                chunks.append(
                    "\n\nEvidence: "
                    + ", ".join(f"`{h[:16]}…`" for h in body.receipt_hashes)
                    + " — verify with `dipcatcher verify-research`."
                )
            return (
                chunks,
                model_snap if isinstance(model_snap, str) else None,
                (time.monotonic() - t0) * 1000.0,
                cast("dict[str, int]", dict(usage_snap)) if isinstance(usage_snap, dict) else None,
                cid,
            )

        def _events(
            chunks: list[str],
            model_name: str | None,
            latency_ms: float,
            usage: dict[str, int] | None = None,
            completion_id: str | None = None,
        ) -> Iterator[str]:
            for chunk in chunks:
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "final",
                        "model": model_name,
                        "receipt_hashes": body.receipt_hashes or [],
                        "latency_ms": latency_ms,
                        "usage": usage,
                        "completion_id": completion_id,
                        "sampling": sampling_fields,
                    }
                )
                + "\n\n"
            )
            yield "data: [DONE]\n\n"

        if sse_keepalive_s <= 0:
            chunks, model_name, latency_ms, usage, cid = _gather()
            return StreamingResponse(
                _events(chunks, model_name, latency_ms, usage, cid),
                media_type="text/event-stream",
            )

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
            chunks, model_name, latency_ms, usage, cid = payload
            return StreamingResponse(
                _events(chunks, model_name, latency_ms, usage, cid),
                media_type="text/event-stream",
            )

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
                chunks, model_name, latency_ms, usage, cid = payload
                yield from _events(chunks, model_name, latency_ms, usage, cid)
                return

        return StreamingResponse(_events_keepalived(), media_type="text/event-stream")

    # --- OpenAI-compatible ingress ----------------------------------------
    # Drop-in surface for stock OpenAI clients: translate the request onto
    # the gated `complete` chain (same resolve/fallback/gate/meter/log path
    # — no second pipeline), then emit the OpenAI envelope. Errors under
    # /v1 get the OpenAI error shape via the app's exception handlers and
    # auth middleware (path-prefix check). Streaming emits gated text as
    # deltas after the honesty gate — first byte arrives post-gate, so
    # first-token latency equals full latency; no ungated bytes ever ship.

    _openai_created = int(time.time())

    @app.get(
        "/v1/models",
        response_model=OpenAIModelList,
        tags=["openai"],
        operation_id="openai_list_models",
    )
    def openai_models() -> OpenAIModelList:
        """Model inventory — the backend names a `model` field may carry,
        plus the `fx1` alias for the default link (hosted_k3)."""
        return OpenAIModelList(
            data=[OpenAIModel(id=m, created=_openai_created) for m in OPENAI_MODEL_IDS]
        )

    @app.post(
        "/v1/chat/completions",
        # the JSON path returns OpenAIChatResponse; stream=true returns SSE
        # chunks (response_model stays None — the route returns a Response)
        responses={200: {"model": OpenAIChatResponse}},
        tags=["openai"],
        operation_id="openai_chat_completions",
    )
    def openai_chat_completions(
        body: OpenAIChatRequest,
        request: Request,
        _slot_held: None = Depends(slot),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> Response:
        """OpenAI-compatible chat completion over the gated pipeline.

        The `X-Fx1-Completion-Id` header links the response to the
        completion-log record (GET /harness/completions/{id}) and its
        sealed receipt. `model` selects a backend when it names one
        (hosted_k3/local_fx1/byok); anything else is the default link.
        BYOK binds via `X-Fx1-Byok-{Base-Url,Api-Key,Model}` headers or the
        `fx1` extension object; `X-Fx1-Backend`/`X-Fx1-Fallbacks`/
        `X-Fx1-Checkpoint-Dir` select the chain without body extensions.
        `Idempotency-Key` makes the call retry-safe: a retry with the
        same key and body replays the stored response byte-identically
        (JSON or SSE) instead of re-spending the model — flagged via
        `X-Fx1-Idempotent-Replay`; a key reused with a different body
        fails closed 409.
        """
        body_fp = _body_fp(body)
        key, replay = _idem_lookup(idempotency_key, openai_idem_store, body_fp)
        if replay is not None:
            env = replay.envelope
            cid_replay = str(env["id"]).removeprefix("chatcmpl-")
            headers = {
                "X-Fx1-Completion-Id": cid_replay,
                "X-Fx1-Idempotent-Replay": "true",
            }
            if body.stream:
                return StreamingResponse(
                    _openai_sse(
                        body,
                        content=env["choices"][0]["message"]["content"],
                        backend=env["system_fingerprint"],
                        model=env["model"],
                        usage=env["usage"],
                        cid=cid_replay,
                        created=env["created"],
                    ),
                    media_type="text/event-stream",
                    headers=headers,
                )
            return JSONResponse(env, headers=headers)
        try:
            creq = CompleteRequest(**openai_to_kwargs(body, request.headers))
        except OpenAICompatError as exc:
            raise ApiError(exc.status, str(exc)) from exc
        out = complete(
            body=creq,
            response=Response(),
            _slot_held=None,
            idempotency_key=None,
        )
        cid = out.completion_id or uuid.uuid4().hex
        envelope = openai_envelope(
            cid=cid,
            content=out.content,
            backend=out.backend,
            model=out.model,
            usage=out.usage,
        )
        if key is not None:
            openai_idem_store.put(key, body_fp, _OpenAIIdemRecord(envelope=envelope))
        if body.stream:
            headers = {"X-Fx1-Completion-Id": cid}
            return StreamingResponse(
                _openai_sse(
                    body,
                    content=out.content,
                    backend=out.backend,
                    model=out.model,
                    usage=out.usage,
                    cid=cid,
                ),
                media_type="text/event-stream",
                headers=headers,
            )
        return JSONResponse(envelope, headers={"X-Fx1-Completion-Id": cid})

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
        body_fp = _body_fp(body)
        key, replay = _idem_lookup(idempotency_key, complete_batch_idem_store, body_fp)
        if replay is not None:
            return replay
        _check_citations(body.receipt_hashes)
        # Resolve-level fallback: first resolvable link serves the whole
        # batch — a shared backend can't attribute per-item usage, so
        # per-item failover is intentionally not offered.
        serving, backend, batch_attempts = _resolve_chain(body)
        sampling_params = _sampling_of(body)
        sampling_fields = sampling_params.body_fields()
        usage_pre = getattr(backend, "total_usage", None)
        usage_pre = dict(usage_pre) if isinstance(usage_pre, dict) else None
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
                    t0 = time.monotonic()
                    cid = uuid.uuid4().hex
                    prompt_sha256 = hashlib.sha256(
                        json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    ).hexdigest()
                    model_snap = getattr(backend, "_model", None)

                    def _log_item(
                        ok: bool, content: str | None, err: str | None, cls: str | None
                    ) -> None:
                        completion_log.append(
                            CompletionRecord(
                                completion_id=cid,
                                backend=serving,
                                model=model_snap if isinstance(model_snap, str) else None,
                                ok=ok,
                                latency_ms=(time.monotonic() - t0) * 1000.0,
                                at=time.time(),
                                # usage is only honest at batch level (shared
                                # endpoint attribution) — per-item stays None.
                                error=err,
                                error_class=cls,
                                prompt_sha256=prompt_sha256,
                                sampling=sampling_fields,
                                output_sha256=(
                                    hashlib.sha256(content.encode("utf-8")).hexdigest()
                                    if content is not None
                                    else None
                                ),
                            )
                        )

                    serving_key = _breaker_key_name(serving, body.byok)
                    if breaker is not None and breaker.check(serving_key) > 0:
                        _log_item(False, None, "backend circuit open", "backend_unavailable")
                        return CompleteBatchItem(
                            ok=False,
                            latency_ms=0.0,
                            error="backend circuit open",
                            error_class="backend_unavailable",
                            completion_id=cid,
                        )
                    item_ok = False
                    try:
                        content = cited_complete(
                            backend,
                            messages,
                            receipt_hashes=body.receipt_hashes,
                            sampling=sampling_params,
                        )
                        item_ok = True
                    except Fx1HonestyError as exc:
                        _log_item(False, None, str(exc), "honesty_refusal")
                        return CompleteBatchItem(
                            ok=False,
                            latency_ms=(time.monotonic() - t0) * 1000.0,
                            error=str(exc),
                            error_class="honesty_refusal",
                            completion_id=cid,
                        )
                    except (
                        BackendNotConfiguredError,
                        RuntimeError,
                        ValueError,
                    ) as exc:
                        if breaker is not None and not isinstance(exc, NotImplementedError):
                            breaker.report(serving_key, False)
                        _log_item(False, None, str(exc), type(exc).__name__)
                        return CompleteBatchItem(
                            ok=False,
                            latency_ms=(time.monotonic() - t0) * 1000.0,
                            error=str(exc),
                            error_class=type(exc).__name__,
                            completion_id=cid,
                        )
                    finally:
                        metrics.record_complete(
                            serving,
                            item_ok,
                            (time.monotonic() - t0) * 1000.0,
                            usage=getattr(backend, "last_usage", None)
                            if isinstance(getattr(backend, "last_usage", None), dict)
                            else None,
                        )
                    _log_item(True, content, None, None)
                    if breaker is not None:
                        breaker.report(serving_key, True)
                    return CompleteBatchItem(
                        ok=True,
                        latency_ms=(time.monotonic() - t0) * 1000.0,
                        content=content,
                        completion_id=cid,
                    )

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
        usage_post = getattr(backend, "total_usage", None)
        if isinstance(usage_post, dict) and usage_post:
            usage_total = {
                k: v - (usage_pre.get(k, 0) if usage_pre else 0) for k, v in usage_post.items()
            }
        else:
            usage_total = None
        resp = CompleteBatchResponse(
            backend=serving,
            model=model_name if isinstance(model_name, str) else None,
            receipt_hashes=body.receipt_hashes or [],
            results=results,
            sampling=sampling_fields,
            usage_total=usage_total,
            attempts=batch_attempts if len(batch_attempts) > 1 else [],
        )
        if key is not None:
            complete_batch_idem_store.put(key, body_fp, resp)
        return resp

    @app.post(
        "/harness/backends/{name}/probe",
        response_model=BackendProbeResponse,
        tags=["ops"],
        operation_id="backend_probe",
    )
    def backend_probe(
        name: Literal["hosted_k3", "local_fx1", "byok"],
        body: BackendProbeRequest | None = None,
        _slot_held: None = Depends(slot),
    ) -> BackendProbeResponse:
        """Deep health: run one minimal gated completion through the real
        resolver. Unlike ``GET /harness/backends`` (config + circuit state),
        this answers "can this backend serve right now" — including a BYOK
        endpoint supplied inline. Deliberately bypasses the breaker admit
        and never reports to it, so a monitoring scrape can't trip or heal
        the circuit; the verdict series lands under ``probe:<name>``."""
        req = body or BackendProbeRequest()
        t0 = time.monotonic()

        def _verdict(resp: BackendProbeResponse) -> BackendProbeResponse:
            with probe_lock:
                probe_cache[name] = BackendProbeVerdict(
                    ok=resp.ok,
                    latency_ms=resp.latency_ms,
                    checked_at=time.time(),
                    error_class=resp.error_class,
                )
            return resp

        try:
            backend = resolve_backend(
                name,
                req.checkpoint_dir,
                req.byok.model_dump() if req.byok is not None else None,
                req.timeout_s if req.timeout_s is not None else 30.0,
            )
        except ApiError as exc:
            # An unconfigured/unreachable backend is a verdict, not an
            # HTTP fault — report it as ok:false. Client-side arg errors
            # (404/422) still propagate as request errors.
            if exc.status_code == 503:
                return _verdict(
                    BackendProbeResponse(
                        backend=name,
                        ok=False,
                        model=None,
                        latency_ms=(time.monotonic() - t0) * 1000.0,
                        error=str(exc.detail),
                        error_class="backend_unavailable",
                    )
                )
            raise
        ok = False
        error: str | None = None
        error_class: str | None = None
        try:
            content = backend.complete([{"role": "user", "content": req.prompt}])
            try:
                validate_fx1_output(content)
            except Fx1HonestyError as exc:
                error, error_class = str(exc), "honesty_refusal"
            else:
                ok = True
        except NotImplementedError as exc:
            error, error_class = str(exc), "NotImplementedError"
        except (BackendNotConfiguredError, RuntimeError, ValueError) as exc:
            error, error_class = str(exc), type(exc).__name__
        finally:
            metrics.record_complete(
                f"probe:{name}",
                ok,
                (time.monotonic() - t0) * 1000.0,
                usage=getattr(backend, "last_usage", None)
                if isinstance(getattr(backend, "last_usage", None), dict)
                else None,
            )
            _close_backend(backend)
        model_name = getattr(backend, "_model", None)
        return _verdict(
            BackendProbeResponse(
                backend=name,
                ok=ok,
                model=model_name if isinstance(model_name, str) else None,
                latency_ms=(time.monotonic() - t0) * 1000.0,
                error=error,
                error_class=error_class,
            )
        )

    @app.post(
        "/harness/gate/check",
        response_model=GateCheckResponse,
        tags=["ops"],
        operation_id="gate_check",
    )
    async def gate_check(body: GateCheckRequest) -> GateCheckResponse:
        """Pre-flight the honesty gate without spending model tokens —
        writers (fx-1 or BYOK callers) can validate text before or after
        generation. Advisory: not slot-gated, stays up during drain, and
        never touches a backend or the metrics series."""
        try:
            validate_fx1_output(body.text)
        except Fx1HonestyError as exc:
            return GateCheckResponse(ok=False, error=str(exc))
        return GateCheckResponse(ok=True)


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
    receipts_dir: str | os.PathLike[str] | None = None,
    byok_override: bool | None = None,
) -> FastAPI:
    api_key = os.environ.get(_API_KEY_ENV) or None
    lab = harness or Harness()
    resolve_backend = backend_resolver or get_backend
    max_inflight = _env_int_bound(_MAX_INFLIGHT_ENV, 16, max_inflight)
    idem_max = _env_int_bound(_IDEM_MAX_ENV, 1024, idem_max)
    job_max = _env_int_bound(_JOB_MAX_ENV, 1024, job_max)
    sse_keepalive_s = _env_float_floor(_SSE_KEEPALIVE_ENV, 15.0, sse_keepalive_s)
    rate_limit_rps = _env_float_floor(_RATE_LIMIT_ENV, 0.0, rate_limit_rps)
    byok_override_enabled = (
        os.environ.get(_BYOK_OVERRIDE_ENV, "1").strip().lower() in ("1", "true", "yes", "on")
        if byok_override is None
        else byok_override
    )
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
    receipts_root = Path(
        receipts_dir if receipts_dir is not None else os.environ.get(_RECEIPTS_DIR_ENV, "receipts")
    )
    receipt_index = _ReceiptIndex(receipts_root)
    # Bounded in-flight work: the harness executes lab commands and model
    # calls on shared resources (a spawned local engine, GPU memory, the
    # box itself) — saturation must fail honestly as 503, never queue
    # unboundedly or crash mid-request. Cheap routes (commands, verify,
    # health) stay uncapped so liveness answers under load.
    inflight = threading.BoundedSemaphore(max_inflight)
    metrics = _Metrics(max_inflight)
    # Last deep-health verdict per backend — process-local so monitoring
    # scrapes read it off ``GET /harness/backends`` without re-probing.
    probe_cache: dict[str, BackendProbeVerdict] = {}
    probe_lock = threading.Lock()
    completion_log = _CompletionLog()
    idem_store: _IdemStore[HarnessRunResponse] = _IdemStore(idem_max)
    complete_idem_store: _IdemStore[CompleteResponse] = _IdemStore(idem_max)
    complete_batch_idem_store: _IdemStore[CompleteBatchResponse] = _IdemStore(idem_max)
    openai_idem_store: _IdemStore[_OpenAIIdemRecord] = _IdemStore(idem_max)
    job_store = _JobStore(job_max)
    eval_store = EvalStore(job_max)
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
        lifespan=_make_lifespan(metrics, job_store, eval_store, jobs_executor),
        openapi_tags=[
            {"name": "runs", "description": "Synchronous lab-command execution."},
            {"name": "jobs", "description": "Async run jobs: submit, poll, SSE, cancel, batch."},
            {
                "name": "complete",
                "description": "Gated model completion (sync, batch, SSE stream).",
            },
            {
                "name": "evals",
                "description": "Async eval-suite submissions against any backend.",
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
    app.state.eval_store = eval_store
    app.state.jobs_executor = jobs_executor
    app.state.sse_keepalive_s = sse_keepalive_s
    app.state.rate_limiter = limiter
    app.state.breaker = breaker

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
        if is_openai_path(request.url.path):
            return JSONResponse(
                status_code=exc.status_code,
                content=openai_error_body(str(exc.detail), exc.status_code, _err_code(exc)),
                headers=exc.headers,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": _err_code(exc)},
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        if is_openai_path(request.url.path):
            errs = jsonable_encoder(exc.errors())
            msg = "; ".join(
                f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg', '')}"
                for e in errs[:4]
            )
            return JSONResponse(
                status_code=422,
                content=openai_error_body(msg or "invalid request", 422, "validation"),
            )
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
        # Public paths are exempt — a load balancer's /health probe cadence
        # must never consume the client's own request budget (or 429 liveness).
        if limiter is not None and request.url.path not in _PUBLIC_PATHS:
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
                rl_msg = f"rate limit exceeded; retry in {wait:.1f}s"
                rl_content: dict[str, Any] = {
                    "detail": rl_msg,
                    "code": "too_many_requests",
                }
                if is_openai_path(request.url.path):
                    rl_content = openai_error_body(rl_msg, 429, "too_many_requests")
                response = JSONResponse(
                    status_code=429,
                    content=rl_content,
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
                    big_content: dict[str, Any] = {
                        "detail": f"body exceeds {_MAX_BODY_BYTES}-byte cap",
                        "code": "too_large",
                    }
                    if is_openai_path(request.url.path):
                        big_content = openai_error_body(
                            str(big_content["detail"]), 413, "too_large"
                        )
                    response = JSONResponse(status_code=413, content=big_content)
                    return _finish(request, request_id, response, started)
        if request.url.path in _PUBLIC_PATHS:
            response = await call_next(request)
        elif api_key:
            provided = request.headers.get("X-API-Key")
            # OpenAI-shape clients authenticate with Authorization: Bearer
            # — accept it on /v1 so stock SDKs work unmodified.
            if not provided and is_openai_path(request.url.path):
                auth_hdr = request.headers.get("Authorization", "")
                if auth_hdr.startswith("Bearer "):
                    provided = auth_hdr[len("Bearer ") :]
            if not provided or not hmac.compare_digest(provided, api_key):
                content: dict[str, Any] = {
                    "detail": "invalid or missing X-API-Key",
                    "code": "unauthorized",
                }
                if is_openai_path(request.url.path):
                    content = openai_error_body("invalid or missing API key", 401, "unauthorized")
                response = JSONResponse(status_code=401, content=content)
            else:
                response = await call_next(request)
        else:
            host = (request.client.host if request.client else "") or ""
            if host not in _LOOPBACK_HOSTS:
                forbidden_body: dict[str, Any] = {
                    "detail": (
                        "FX1_API_KEY is unset; non-localhost clients are "
                        "refused. Set FX1_API_KEY and send X-API-Key, or "
                        "bind to 127.0.0.1 only."
                    ),
                    "code": "forbidden",
                }
                if is_openai_path(request.url.path):
                    forbidden_body = openai_error_body(
                        str(forbidden_body["detail"]), 403, "forbidden"
                    )
                response = JSONResponse(status_code=403, content=forbidden_body)
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
                "receipts_store": receipt_index.available(),
                "byok_override": byok_override_enabled,
                "openai_compat": True,
                "evals": True,
            },
            eval_suites=list(EVAL_SUITES),
            limits={
                "max_inflight": float(metrics.max_inflight),
                "job_max": float(job_store._max),
                "eval_max": float(eval_store.capacity),
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
            with probe_lock:
                last_probe = probe_cache.get(name)
            out[name] = BackendStatusEntry(
                configured=cfg,
                circuit_open=circuit_open,
                cooldown_remaining_s=round(remaining, 3),
                consecutive_failures=fails,
                last_probe=last_probe,
            )
        return out

    @app.get(
        "/harness/completions",
        response_model=CompletionListResponse,
        tags=["ops"],
        operation_id="completions_list",
    )
    def completions_list(
        limit: int = Query(default=50, ge=1, le=_COMPLETION_LOG_MAX),
        backend: Literal["hosted_k3", "local_fx1", "byok"] | None = None,
    ) -> CompletionListResponse:
        """Newest-first window on the completion log — per-call evidence
        (hashes, usage, verdict) for every gated model call the process
        has served, bounded by the ring."""
        items = completion_log.latest(limit, backend)
        return CompletionListResponse(items=items, count=len(items))

    @app.get(
        "/harness/completions/{completion_id}",
        response_model=CompletionRecord,
        tags=["ops"],
        operation_id="completion_get",
    )
    def completion_get(completion_id: str) -> CompletionRecord:
        rec = completion_log.get(completion_id)
        if rec is None:
            raise ApiError(404, f"completion {completion_id!r} not in the log", code="not_found")
        return rec

    @app.get(
        "/harness/completions/{completion_id}/receipt",
        tags=["ops"],
        operation_id="completion_receipt",
    )
    def completion_receipt(completion_id: str) -> dict[str, Any]:
        """Export one logged call as a sealed ``fx1_completion_record.v1``
        document — verify with ``POST /receipts/verify`` or the SDK."""
        from fx1.serve.ops_receipt import (  # noqa: PLC0415
            completion_record_receipt,
        )

        rec = completion_log.get(completion_id)
        if rec is None:
            raise ApiError(404, f"completion {completion_id!r} not in the log", code="not_found")
        return completion_record_receipt(rec.model_dump(mode="json"))

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
        body_fp = _body_fp(body, exclude={"idempotency_key"})
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

    def _resolve_request_backend(
        backend_name: str,
        checkpoint_dir: str | None,
        byok: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        """Checkpoint validation + backend resolution → HTTP error map."""
        kwargs: dict[str, Any] = {}
        if timeout_s is not None:
            kwargs["timeout_s"] = timeout_s
        if byok is not None:
            if backend_name != "byok":
                raise ApiError(422, "a byok override applies only to backend='byok'")
            if not byok_override_enabled:
                raise ApiError(
                    422,
                    "per-request BYOK credentials are disabled on this server "
                    f"({_BYOK_OVERRIDE_ENV}=0 / byok_override=False)",
                    code="byok_override_disabled",
                )
            kwargs.update(byok)
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
        openai_idem_store=openai_idem_store,
        breaker=breaker,
        receipt_index=receipt_index,
        metrics=metrics,
        probe_cache=probe_cache,
        probe_lock=probe_lock,
        completion_log=completion_log,
        eval_store=eval_store,
        inflight=inflight,
        jobs_executor=jobs_executor,
    )

    _mount_receipt_routes(app, receipt_index)

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
