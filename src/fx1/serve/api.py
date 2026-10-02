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

import hmac
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

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
from quant_fund.research.receipt_v2 import verify_receipt_payload

_API_KEY_ENV = "FX1_API_KEY"
_MAX_INFLIGHT_ENV = "FX1_API_MAX_INFLIGHT"
_SSE_KEEPALIVE_ENV = "FX1_API_SSE_KEEPALIVE_S"
_IDEM_MAX_ENV = "FX1_API_IDEM_MAX"
_IDEM_KEY_MAX = 256
_JOB_MAX_ENV = "FX1_API_JOB_MAX"
_PUBLIC_PATHS = frozenset({"/health"})
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})
_MAX_BODY_BYTES = 1 << 20


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


class HarnessRunResponse(_Model):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    ok: bool
    timeout_s: int
    replayed: bool = False


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


class DrainResponse(_Model):
    """Result of latching drain mode: state + live in-flight count."""

    draining: bool
    inflight: int
    drained: bool = False


class ReadyResponse(_Model):
    """Readiness probe payload — only emitted while accepting work."""

    ready: Literal[True] = True
    inflight: int


class JobSubmitResponse(_Model):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    replayed: bool = False


class JobStatusResponse(_Model):
    """Live job record; ``result`` appears only once status is terminal."""

    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: float
    finished_at: float | None
    result: HarnessRunResponse | None
    error: str | None


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
        self.draining = threading.Event()

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
            )


def _env_int_bound(name: str, default: int, given: int | None) -> int:
    """Positive-int tunable from arg or env, fail-closed below 1."""
    v = int(os.environ.get(name, str(default))) if given is None else given
    if v < 1:
        raise ValueError(f"{name} bound must be >= 1, got {v}")
    return v


def _idem_lookup(
    idempotency_key: str | None,
    store: _IdemStore,
    body_fp: str,
) -> tuple[str | None, HarnessRunResponse | None]:
    """Shared Idempotency-Key preamble: normalize + bound the key, then
    look up a stored replay. Returns ``(key, cached)`` — a cached hit
    is the response to return verbatim plus ``replayed: True``; a key
    reused with a different body fails closed 409."""
    key = (idempotency_key or "").strip() or None
    if key is None:
        return None, None
    if len(key) > _IDEM_KEY_MAX:
        raise HTTPException(400, "Idempotency-Key must be <= 256 chars")
    entry = store.get(key)
    if entry is None:
        return key, None
    fp, cached = entry
    if fp != body_fp:
        raise HTTPException(409, "Idempotency-Key reuse with a different request body")
    return key, cached.model_copy(update={"replayed": True})


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
        raise HTTPException(400, "Idempotency-Key must be <= 256 chars")
    body_fp = body.model_dump_json()
    if key is not None:
        entry = job_store.get_key(key)
        if entry is not None:
            fp, job_id = entry
            if fp != body_fp:
                raise HTTPException(
                    409,
                    "Idempotency-Key reuse with a different request body",
                )
            job = job_store.get(job_id)
            if job is not None:
                return JobSubmitResponse(job_id=job_id, status=job.status, replayed=True)
    try:
        lab.get(body.command)  # fail closed at submit, not in the worker
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if metrics.draining.is_set():
        raise HTTPException(503, "harness is draining — no new work accepted")
    if not inflight.acquire(blocking=False):
        raise HTTPException(
            503,
            f"harness at max_inflight={metrics.max_inflight} — retry later",
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
    )

    def _exec() -> None:
        job.status = "running"
        try:
            result = lab.run(
                body.command,
                body.extra_args or None,
                config=Path(body.config) if body.config else None,
            )
            command = lab.get(body.command)
            job.result = HarnessRunResponse(
                command=result.command,
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                ok=result.ok,
                timeout_s=command.timeout_s,
            )
            job.status = "succeeded"
        except Exception as exc:  # noqa: BLE001 — worker faults land in the record
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = "failed"
        job.finished_at = time.time()
        metrics.release()
        inflight.release()

    try:
        jobs_executor.submit(_exec)
    except RuntimeError as exc:  # executor gone (shutdown race)
        metrics.release()
        inflight.release()
        raise HTTPException(503, "job executor unavailable") from exc
    job_store.put(job, key, body_fp)
    return JobSubmitResponse(job_id=job.job_id, status=job.status, replayed=False)


class _IdemStore:
    """Bounded LRU of ``Idempotency-Key`` -> run response.

    Lets a client (or the HarnessClient, which mints a key per ``run``)
    retry a submission after a transport blip without double-executing
    the command. Read-only replays bypass the drain latch and the
    concurrency cap: the work already happened.
    """

    def __init__(self, max_entries: int) -> None:
        self._lock = threading.Lock()
        self._max = max_entries
        self._map: OrderedDict[str, tuple[str, HarnessRunResponse]] = OrderedDict()

    def get(self, key: str) -> tuple[str, HarnessRunResponse] | None:
        with self._lock:
            hit = self._map.get(key)
            if hit is not None:
                self._map.move_to_end(key)
            return hit

    def put(self, key: str, fingerprint: str, resp: HarnessRunResponse) -> None:
        with self._lock:
            self._map[key] = (fingerprint, resp)
            self._map.move_to_end(key)
            while len(self._map) > self._max:
                self._map.popitem(last=False)


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


def create_app(
    harness: Harness | None = None,
    backend_resolver: Any | None = None,
    max_inflight: int | None = None,
    sse_keepalive_s: float | None = None,
    idem_max: int | None = None,
    job_max: int | None = None,
) -> FastAPI:
    api_key = os.environ.get(_API_KEY_ENV) or None
    lab = harness or Harness()
    resolve_backend = backend_resolver or get_backend
    max_inflight = _env_int_bound(_MAX_INFLIGHT_ENV, 16, max_inflight)
    idem_max = _env_int_bound(_IDEM_MAX_ENV, 1024, idem_max)
    job_max = _env_int_bound(_JOB_MAX_ENV, 1024, job_max)
    if sse_keepalive_s is None:
        sse_keepalive_s = float(os.environ.get(_SSE_KEEPALIVE_ENV, "15"))
    if sse_keepalive_s < 0:
        raise ValueError(f"sse_keepalive_s must be >= 0, got {sse_keepalive_s}")
    # Bounded in-flight work: the harness executes lab commands and model
    # calls on shared resources (a spawned local engine, GPU memory, the
    # box itself) — saturation must fail honestly as 503, never queue
    # unboundedly or crash mid-request. Cheap routes (commands, verify,
    # health) stay uncapped so liveness answers under load.
    inflight = threading.BoundedSemaphore(max_inflight)
    metrics = _Metrics(max_inflight)
    idem_store = _IdemStore(idem_max)
    job_store = _JobStore(job_max)
    jobs_executor = ThreadPoolExecutor(max_workers=max_inflight, thread_name_prefix="fx1-job")

    @contextmanager
    def _work_gate() -> Iterator[None]:
        if metrics.draining.is_set():
            raise HTTPException(
                503,
                "harness is draining — no new work accepted",
            )
        if not inflight.acquire(blocking=False):
            raise HTTPException(
                503,
                f"harness at max_inflight={max_inflight} — retry later",
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
    )
    app.state.inflight_slots = inflight
    app.state.metrics = metrics
    app.state.idem_store = idem_store
    app.state.job_store = job_store
    app.state.jobs_executor = jobs_executor
    app.state.sse_keepalive_s = sse_keepalive_s

    @app.middleware("http")
    async def harness_api_auth(request: Request, call_next: Any) -> Any:
        request_id = _request_id(request.headers.get("x-request-id"))
        request.state.request_id = request_id
        started = time.monotonic()
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            declared = request.headers.get("content-length")
            if declared is not None:
                try:
                    length = int(declared)
                except ValueError:
                    response = JSONResponse(
                        status_code=400, content={"detail": "invalid content-length"}
                    )
                    return _finish(request, request_id, response, started)
                if length > _MAX_BODY_BYTES:
                    response = JSONResponse(
                        status_code=413,
                        content={"detail": f"body exceeds {_MAX_BODY_BYTES}-byte cap"},
                    )
                    return _finish(request, request_id, response, started)
        if request.url.path in _PUBLIC_PATHS:
            response = await call_next(request)
        elif api_key:
            provided = request.headers.get("X-API-Key")
            if not provided or not hmac.compare_digest(provided, api_key):
                response = JSONResponse(
                    status_code=401, content={"detail": "invalid or missing X-API-Key"}
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
                        )
                    },
                )
            else:
                response = await call_next(request)
        return _finish(request, request_id, response, started)

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics_route() -> MetricsResponse:
        return metrics.snapshot()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            registered_commands=len(lab.list_commands()),
            backends=_backend_configured(),
            draining=metrics.draining.is_set(),
        )

    @app.get("/ready", response_model=ReadyResponse)
    def ready() -> ReadyResponse:
        """Kubernetes-style readiness: 200 while accepting work, 503 once
        drain is latched — the load balancer's signal to deregister the
        pod before gated routes start refusing."""
        if metrics.draining.is_set():
            raise HTTPException(503, "harness is draining")
        return ReadyResponse(inflight=metrics.snapshot().inflight)

    @app.post("/harness/drain", response_model=DrainResponse)
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

    @app.get("/harness/commands", response_model=HarnessCommandListResponse)
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

    @app.post("/harness/runs", response_model=HarnessRunResponse)
    def run_command(
        body: HarnessRunRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> HarnessRunResponse:
        body_fp = body.model_dump_json()
        key, replay = _idem_lookup(idempotency_key, idem_store, body_fp)
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
                raise HTTPException(404, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
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

    @app.post("/harness/jobs", response_model=JobSubmitResponse, status_code=202)
    def submit_job(
        body: HarnessRunRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JobSubmitResponse:
        """Async run submission: work starts in the background, the caller
        polls ``GET /harness/jobs/{job_id}`` for the terminal record.
        Same drain/cap/idempotency contract as the sync route."""
        return _submit_job(body, idempotency_key, lab, job_store, metrics, inflight, jobs_executor)

    @app.get("/harness/jobs/{job_id}", response_model=JobStatusResponse)
    def job_status(job_id: str) -> JobStatusResponse:
        job = job_store.get(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job_id {job_id!r}")
        return job

    def _resolve_request_backend(backend_name: str, checkpoint_dir: str | None) -> Any:
        """Checkpoint validation + backend resolution → HTTP error map."""
        kwargs: dict[str, Any] = {}
        if backend_name == "local_fx1":
            checkpoint = checkpoint_dir or os.environ.get("FX1_CHECKPOINT_DIR")
            if not checkpoint:
                raise HTTPException(
                    422,
                    "local_fx1 needs a checkpoint_dir in the request or "
                    "FX1_CHECKPOINT_DIR on the server",
                )
            kwargs["checkpoint_dir"] = checkpoint
        elif checkpoint_dir is not None:
            raise HTTPException(422, "checkpoint_dir applies only to the local_fx1 backend")
        try:
            return resolve_backend(backend_name, **kwargs)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            # Missing credentials / unsigned release / failed ship gate are
            # server-side configuration faults, not client input errors.
            raise HTTPException(503, str(exc)) from exc

    def _close_backend(backend: Any) -> None:
        closer = getattr(backend, "close", None)
        if callable(closer):
            closer()

    @app.post("/harness/complete", response_model=CompleteResponse)
    def complete(body: CompleteRequest, _slot_held: None = Depends(_slot)) -> CompleteResponse:
        backend = _resolve_request_backend(body.backend, body.checkpoint_dir)
        messages = [{"role": m.role, "content": m.content} for m in body.messages]
        try:
            content = cited_complete(backend, messages, receipt_hashes=body.receipt_hashes)
        except BackendNotConfiguredError as exc:
            raise HTTPException(503, str(exc)) from exc
        except NotImplementedError as exc:
            raise HTTPException(501, str(exc)) from exc
        except Fx1HonestyError as exc:
            # The model produced a contract-violating headline; the gate
            # caught it before the bytes left — surface as 502, not success.
            raise HTTPException(502, f"honesty gate refused model output: {exc}") from exc
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from exc
        finally:
            _close_backend(backend)
        model_name = getattr(backend, "_model", None)
        return CompleteResponse(
            backend=body.backend,
            model=model_name if isinstance(model_name, str) else None,
            content=content,
            receipt_hashes=body.receipt_hashes or [],
        )

    @app.post("/harness/complete/stream")
    def complete_stream(
        body: CompleteRequest, _slot_held: None = Depends(_slot)
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
            backend = _resolve_request_backend(body.backend, body.checkpoint_dir)
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
                    raise HTTPException(502, f"honesty gate refused model output: {exc}") from exc
            except BackendNotConfiguredError as exc:
                raise HTTPException(503, str(exc)) from exc
            except NotImplementedError as exc:
                raise HTTPException(501, str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(502, str(exc)) from exc
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

    @app.post("/harness/complete/batch", response_model=CompleteBatchResponse)
    def complete_batch(
        body: CompleteBatchRequest, _slot_held: None = Depends(_slot)
    ) -> CompleteBatchResponse:
        backend = _resolve_request_backend(body.backend, body.checkpoint_dir)
        # One backend serves the whole batch — a spawned local engine is
        # shared across workers (spawn path is lock-guarded). Item failures
        # are per-slot verdicts: a gate refusal on one prompt does not lose
        # the rest of the batch.
        try:
            with ThreadPoolExecutor(
                max_workers=min(body.max_workers, len(body.batch)),
                thread_name_prefix="fx1-complete",
            ) as pool:

                def _one(messages: list[dict[str, str]]) -> CompleteBatchItem:
                    try:
                        return CompleteBatchItem(
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
                        NotImplementedError,
                        RuntimeError,
                        ValueError,
                    ) as exc:
                        return CompleteBatchItem(
                            ok=False, error=str(exc), error_class=type(exc).__name__
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
            closer = getattr(backend, "close", None)
            if callable(closer):
                closer()
        model_name = getattr(backend, "_model", None)
        return CompleteBatchResponse(
            backend=body.backend,
            model=model_name if isinstance(model_name, str) else None,
            receipt_hashes=body.receipt_hashes or [],
            results=results,
        )

    @app.post("/receipts/verify", response_model=ReceiptVerifyResponse)
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
