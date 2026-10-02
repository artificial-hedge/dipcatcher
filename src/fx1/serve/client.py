"""HarnessClient — the HTTP twin of ``Fx1Harness`` for remote deployments.

``fx1.sdk`` is the in-process surface; this module is the same contract
over the wire for callers that point at a ``fx1 harness serve`` instance —
fx-1 pods, other services, notebooks. It deliberately returns the SDK's own
result types (``CompletionResult``, ``ReceiptVerdict``, ``HarnessHealth``,
``HarnessResult``) so switching between in-process and remote is a
one-line construction change.

Error mapping mirrors the SDK taxonomy (the wire's status codes map back
to the same exception classes the SDK raises):

- 404                        ``KeyError``
- 422                        ``ValueError``
- 503                        ``BackendNotConfiguredError``
- 501                        ``NotImplementedError``
- 502 ``honesty gate``       ``Fx1HonestyError``
- 502 other                  ``RuntimeError``
- 401/403                    ``HarnessAuthError``
- transport / other          ``HarnessTransportError``

``transport`` is injectable: ``(method, url, json_payload, headers,
timeout_s) -> (status, headers, body_bytes)``. Production uses urllib;
tests route it at a ``fastapi.testclient.TestClient``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from fx1 import __version__
from fx1.harness import HarnessResult
from fx1.honesty import Fx1HonestyError
from fx1.sdk import (
    CompletionRecord,
    CompletionResult,
    GateCheckResult,
    HarnessHealth,
    OpsMetrics,
    ProbeResult,
    ReceiptRef,
    ReceiptVerdict,
    StoredReceipt,
)
from fx1.serve.backends import BackendNotConfiguredError
from fx1.serve.contract import API_VERSION as EXPECTED_API_VERSION

__all__ = [
    "EXPECTED_API_VERSION",
    "HarnessAuthError",
    "HarnessClient",
    "HarnessCompatError",
    "HarnessJobError",
    "HarnessTransportError",
]

Transport = Callable[
    [str, str, dict[str, Any] | None, dict[str, str], float],
    "tuple[int, Mapping[str, str], bytes]",
]


class HarnessTransportError(RuntimeError):
    """Network-level failure talking to the harness API."""

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class HarnessAuthError(PermissionError):
    """401/403 from the harness API — key missing or wrong."""

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class HarnessJobError(RuntimeError):
    """An async run job reached its terminal state without a result —
    the worker captured an exception (``status == 'failed'``)."""

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class HarnessCompatError(RuntimeError):
    """The remote harness speaks a wire contract this client can't parse —
    raised by ``check_compat`` when the server's ``api_version`` differs
    from ``EXPECTED_API_VERSION`` (or the peer predates versioning)."""

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def _urllib_transport(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    headers: dict[str, str],
    timeout_s: float,
) -> tuple[int, Mapping[str, str], bytes]:
    req_headers = {"Accept": "application/json", **headers}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 — URL is validated at construction  # nosec B310
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read() if exc.fp is not None else b""
        return exc.code, dict(exc.headers or {}), body
    except urllib.error.URLError as exc:
        raise HarnessTransportError(f"harness unreachable at {url}: {exc.reason}") from exc


def _retry_after_s(headers: Mapping[str, str]) -> float | None:
    """Parse a Retry-After seconds hint; absent/malformed -> None."""
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                return max(0.0, float(value))
            except ValueError:
                return None
    return None


class HarnessClient:
    """Remote harness client — the SDK contract over HTTP."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        transport: Transport | None = None,
        max_retries: int = 0,
        retry_backoff_s: float = 0.1,
        retry_writes: bool = False,
        max_retry_wait_s: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
        circuit_breaker_threshold: int = 0,
        circuit_reset_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"base_url must be http(s)://host[:port], got {base_url!r}")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if retry_backoff_s <= 0:
            raise ValueError(f"retry_backoff_s must be > 0, got {retry_backoff_s}")
        if max_retry_wait_s <= 0:
            raise ValueError(f"max_retry_wait_s must be > 0, got {max_retry_wait_s}")
        if circuit_breaker_threshold < 0:
            raise ValueError(
                f"circuit_breaker_threshold must be >= 0, got {circuit_breaker_threshold}"
            )
        if circuit_reset_s <= 0:
            raise ValueError(f"circuit_reset_s must be > 0, got {circuit_reset_s}")
        self._base = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
        self._headers = {"X-API-Key": api_key} if api_key else {}
        self._timeout_s = timeout_s
        self._transport = transport or _urllib_transport
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self._retry_writes = retry_writes
        self._max_retry_wait_s = max_retry_wait_s
        self._sleep = sleep
        self._cb_threshold = circuit_breaker_threshold
        self._cb_reset_s = circuit_reset_s
        self._clock = clock
        self._last_api_version: str | None = None
        self._cb_failures = 0
        self._cb_open_until = 0.0

    # ---- transport ----------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        idempotent: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[int, Mapping[str, str], bytes]:
        if self._cb_threshold and self._clock() < self._cb_open_until:
            raise HarnessTransportError(
                f"circuit open for {self._base} — fail fast until "
                f"{self._cb_open_until:.1f} (monotonic)"
            )
        retries = self._max_retries if (idempotent or self._retry_writes) else 0
        backoff = self._retry_backoff_s
        try:
            for attempt in range(retries + 1):
                try:
                    status, headers, body = self._transport(
                        method,
                        self._base + path,
                        payload,
                        {**self._headers, **(extra_headers or {})},
                        self._timeout_s,
                    )
                    for hk, hv in headers.items():
                        if hk.lower() == "x-fx1-api-version":
                            self._last_api_version = hv
                except HarnessTransportError:
                    if attempt >= retries:
                        raise
                    self._sleep(backoff)
                    backoff *= 2
                    continue
                if self._retryable_status(status, headers) and attempt < retries:
                    wait = _retry_after_s(headers)
                    if wait is not None and wait > self._max_retry_wait_s:
                        break  # server asked for a wait longer than the budget allows
                    self._sleep(min(wait if wait is not None else backoff, self._max_retry_wait_s))
                    backoff *= 2
                    continue
                if status < 200 or status >= 300:
                    raise self._map_error(status, body)
                self._cb_reset()
                return status, headers, body
            raise HarnessTransportError(f"harness {method} {path} exhausted {retries} retries")
        except HarnessTransportError:
            self._cb_trip()
            raise

    def _cb_reset(self) -> None:
        self._cb_failures = 0
        self._cb_open_until = 0.0

    def _cb_trip(self) -> None:
        """Count a transport fault; past the threshold the circuit opens for
        ``circuit_reset_s`` (the first call after that is the half-open probe —
        a success closes it, a fault re-opens the window)."""
        if not self._cb_threshold:
            return
        self._cb_failures += 1
        if self._cb_failures >= self._cb_threshold:
            self._cb_open_until = self._clock() + self._cb_reset_s
            self._cb_failures = 0

    @staticmethod
    def _map_error(status: int, body: bytes) -> Exception:
        code: str | None = None
        try:
            parsed = json.loads(body)
            detail = parsed.get("detail", body.decode(errors="replace"))
            raw_code = parsed.get("code")
            if isinstance(raw_code, str):
                code = raw_code
            if isinstance(detail, list):  # pydantic validation errors
                detail = "; ".join(
                    d.get("msg", str(d)) for d in detail if isinstance(d, dict)
                ) or str(detail)
        except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
            detail = body.decode(errors="replace")[:500]
        if status in (401, 403):
            return HarnessAuthError(f"harness auth refused ({status}): {detail}", code=code)
        if status == 404:
            return KeyError(str(detail))
        if status == 422:
            return ValueError(str(detail))
        if status == 501:
            return NotImplementedError(str(detail))
        if status == 503:
            return BackendNotConfiguredError(str(detail), code=code)
        if status == 502:
            text = str(detail)
            if "honesty gate" in text:
                return Fx1HonestyError(
                    text.split("honesty gate refused model output: ")[-1],
                    code=code,
                )
            return RuntimeError(text)
        return HarnessTransportError(f"harness API returned {status}: {detail}", code=code)

    @staticmethod
    def _retryable_status(status: int, headers: Mapping[str, str]) -> bool:
        """429 always retries; 503 retries only when it carries Retry-After
        (the in-flight cap — an unconfigured-backend 503 never will be)."""
        return status == 429 or (status == 503 and _retry_after_s(headers) is not None)

    def _json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        idempotent: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> Any:
        _, _, body = self._request(
            method,
            path,
            payload,
            idempotent=idempotent,
            extra_headers=extra_headers,
        )
        return json.loads(body)

    # ---- registry -----------------------------------------------------

    def commands(self, role: str | None = None) -> list[str]:
        """Registered command names, like ``Fx1Harness.commands``."""
        path = "/harness/commands"
        if role is not None:
            path += f"?role={urllib.parse.quote(role)}"
        out = self._json("GET", path, idempotent=True)
        return [item["name"] for item in out["items"]]

    # ---- execution ------------------------------------------------------

    def run(
        self,
        command: str,
        extra_args: list[str] | None = None,
        *,
        config: str | Path | None = None,
        idempotency_key: str | None = None,
    ) -> HarnessResult:
        """Remote counterpart of ``Fx1Harness.run``.

        Carries an ``Idempotency-Key`` (auto-minted unless the caller
        supplies one for cross-process dedup), so a transport-level
        retry returns the stored result instead of re-executing the
        command — ``max_retries``/``retry_writes`` is safe here.
        """
        key = idempotency_key or uuid.uuid4().hex
        out = self._json(
            "POST",
            "/harness/runs",
            {
                "command": command,
                "extra_args": extra_args or [],
                "config": str(config) if config is not None else None,
            },
            idempotent=True,
            extra_headers={"Idempotency-Key": key},
        )
        return HarnessResult(
            command=out["command"],
            exit_code=out["exit_code"],
            stdout=out["stdout"],
            stderr=out["stderr"],
        )

    def submit_run(
        self,
        command: str,
        extra_args: list[str] | None = None,
        *,
        config: str | Path | None = None,
        idempotency_key: str | None = None,
        callback_url: str | None = None,
        callback_secret: str | None = None,
    ) -> str:
        """Submit a run as a background job; returns the job id.

        Long-running commands shouldn't hold a request open — submit,
        keep the id, and poll with ``job_status``/``wait_run``. Same
        ``Idempotency-Key`` dedup as ``run``: a retried submit returns
        the original job id instead of spawning a second execution.
        ``callback_url`` (http(s)) gets the full job record POSTed on
        every terminal transition — succeeded, failed, or cancelled.
        ``callback_secret`` signs the delivery: the POST carries
        ``X-Fx1-Webhook-Timestamp`` + ``X-Fx1-Webhook-Signature``
        (``sha256=<hmac-sha256>`` over ``<ts>.<body>``); verify with
        ``fx1.serve.webhooks.verify_webhook``.
        """
        key = idempotency_key or uuid.uuid4().hex
        out = self._json(
            "POST",
            "/harness/jobs",
            {
                "command": command,
                "extra_args": extra_args or [],
                "config": str(config) if config is not None else None,
                "callback_url": callback_url,
                "callback_secret": callback_secret,
            },
            idempotent=True,
            extra_headers={"Idempotency-Key": key},
        )
        return str(out["job_id"])

    def submit_batch(
        self,
        jobs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """POST /harness/jobs/batch — fan-out submit, one request.

        Each item is a run-request dict (``command``, ``extra_args``,
        ``config``, ``callback_url``, ``callback_secret``); a per-item
        ``idempotency_key`` dedups retries (headers carry no per-item
        keys). Per-item failures land in ``jobs[i].error``/``code`` —
        the response carries ``submitted``/``failed`` counts."""
        out = self._json("POST", "/harness/jobs/batch", {"jobs": jobs})
        return dict(out)

    def job_status(self, job_id: str) -> dict[str, Any]:
        """Live job record: ``status`` in queued/running/succeeded/
        failed/cancelled; ``result`` (the HarnessRunResponse fields)
        appears once terminal."""
        out = self._json(
            "GET",
            f"/harness/jobs/{urllib.parse.quote(job_id)}",
            idempotent=True,
        )
        return dict(out)

    def list_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """GET /harness/jobs — inventory page, newest first; ``total`` is
        the filtered count so callers can page until offset >= total."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status is not None:
            params["status"] = status
        out = self._json(
            "GET",
            "/harness/jobs?" + urllib.parse.urlencode(params),
            idempotent=True,
        )
        return dict(out)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """DELETE /harness/jobs/{job_id} — cooperative cancel: a queued
        job lands 'cancelled' and its slot frees on dequeue; a running
        or terminal job maps the 409 through the error table."""
        out = self._json(
            "DELETE",
            f"/harness/jobs/{urllib.parse.quote(job_id)}",
        )
        return dict(out)

    def wait_run(
        self,
        job_id: str,
        *,
        poll_s: float = 0.5,
        timeout_s: float | None = None,
    ) -> HarnessResult:
        """Poll ``job_status`` until the job finishes; returns its result.

        Raises ``HarnessJobError`` on status 'failed' and
        ``HarnessTransportError`` when ``timeout_s`` elapses — a timed-out
        waiter leaves the job running server-side; keep polling or leave
        it for the record.
        """
        deadline = None if timeout_s is None else self._clock() + timeout_s
        while True:
            st = self.job_status(job_id)
            if st["status"] == "succeeded":
                r = st["result"]
                return HarnessResult(
                    command=r["command"],
                    exit_code=r["exit_code"],
                    stdout=r["stdout"],
                    stderr=r["stderr"],
                )
            if st["status"] == "failed":
                raise HarnessJobError(f"job {job_id} failed: {st.get('error')}")
            if st["status"] == "cancelled":
                raise HarnessJobError(f"job {job_id} cancelled")
            remaining = None if deadline is None else deadline - self._clock()
            if remaining is not None and remaining <= 0:
                raise HarnessTransportError(f"job {job_id} did not finish within {timeout_s}s")
            self._sleep(poll_s if remaining is None else min(poll_s, remaining))

    def stream_job(
        self,
        job_id: str,
        *,
        timeout_s: float = 600.0,
    ) -> list[dict[str, Any]]:
        """Follow a job over SSE — ``GET /harness/jobs/{job_id}/events``
        returns the job record as an ``event: job`` frame on every status
        change until the job goes terminal, then closes. Returns every
        distinct record snapshot in order; the last frame is terminal
        unless the server-side ``timeout_s`` elapses first (reconnect to
        resume — frames carry the full record, not diffs)."""
        _, _, body = self._request(
            "GET",
            f"/harness/jobs/{urllib.parse.quote(job_id)}/events"
            f"?timeout_s={urllib.parse.quote(str(timeout_s))}",
            idempotent=True,
        )
        frames: list[dict[str, Any]] = []
        for line in body.decode().splitlines():
            if not line.startswith("data: "):
                continue
            frames.append(json.loads(line[len("data: ") :]))
        if not frames:
            raise HarnessTransportError("job stream ended without a status frame")
        return frames

    def wait_run_stream(
        self,
        job_id: str,
        *,
        timeout_s: float = 600.0,
    ) -> HarnessResult:
        """``wait_run`` without polling: follows the SSE job stream, then
        maps the terminal record through the same outcome table. A stream
        that ends before a terminal status raises HarnessTransportError
        (the job may still be running server-side — reconnect or poll)."""
        st = self.stream_job(job_id, timeout_s=timeout_s)[-1]
        if st["status"] == "succeeded":
            r = st["result"]
            return HarnessResult(
                command=r["command"],
                exit_code=r["exit_code"],
                stdout=r["stdout"],
                stderr=r["stderr"],
            )
        if st["status"] == "failed":
            raise HarnessJobError(f"job {job_id} failed: {st.get('error')}")
        if st["status"] == "cancelled":
            raise HarnessJobError(f"job {job_id} cancelled")
        raise HarnessTransportError(
            f"job {job_id} stream ended at {st['status']!r} before terminal"
        )

    # ---- gated completion ------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        backend: str = "local_fx1",
        checkpoint_dir: str | Path | None = None,
        receipt_hashes: list[str] | None = None,
        idempotency_key: str | None = None,
        timeout_s: float | None = None,
        byok: dict[str, str] | None = None,
    ) -> CompletionResult:
        """Remote counterpart of ``Fx1Harness.complete``."""
        out = self._json(
            "POST",
            "/harness/complete",
            {
                "backend": backend,
                "messages": messages,
                "checkpoint_dir": (str(checkpoint_dir) if checkpoint_dir is not None else None),
                "receipt_hashes": receipt_hashes,
                "timeout_s": timeout_s,
                "byok": byok,
            },
            # A keyed complete dedupes server-side — safe to retry by
            # construction, so it marks idempotent for the retry policy.
            idempotent=idempotency_key is not None,
            extra_headers={"Idempotency-Key": idempotency_key} if idempotency_key else None,
        )
        return CompletionResult(
            backend=out["backend"],
            model=out["model"],
            content=out["content"],
            receipt_hashes=tuple(out["receipt_hashes"]),
            replayed=out.get("replayed", False),
            usage=out.get("usage") if isinstance(out.get("usage"), dict) else None,
            completion_id=out.get("completion_id"),
        )

    def completion(self, completion_id: str) -> CompletionRecord:
        """Fetch one recorded call from the server's completion log —
        ``GET /harness/completions/{id}``; 404 maps to KeyError."""
        out = self._json("GET", f"/harness/completions/{completion_id}", idempotent=True)
        return CompletionRecord(
            completion_id=out["completion_id"],
            backend=out["backend"],
            model=out.get("model"),
            ok=out["ok"],
            latency_ms=out["latency_ms"],
            at=out["at"],
            usage=out.get("usage") if isinstance(out.get("usage"), dict) else None,
            error=out.get("error"),
            error_class=out.get("error_class"),
            prompt_sha256=out["prompt_sha256"],
            output_sha256=out.get("output_sha256"),
        )

    def completions(self, *, limit: int = 50, backend: str | None = None) -> list[CompletionRecord]:
        """Newest-first window on the server's completion log —
        ``GET /harness/completions``."""
        params: dict[str, Any] = {"limit": limit}
        if backend is not None:
            params["backend"] = backend
        out = self._json(
            "GET",
            f"/harness/completions?{urllib.parse.urlencode(params)}",
            idempotent=True,
        )
        return [
            CompletionRecord(
                completion_id=r["completion_id"],
                backend=r["backend"],
                model=r.get("model"),
                ok=r["ok"],
                latency_ms=r["latency_ms"],
                at=r["at"],
                usage=r.get("usage") if isinstance(r.get("usage"), dict) else None,
                error=r.get("error"),
                error_class=r.get("error_class"),
                prompt_sha256=r["prompt_sha256"],
                output_sha256=r.get("output_sha256"),
            )
            for r in out["items"]
        ]

    def completion_receipt(self, completion_id: str) -> dict[str, Any]:
        """Export one logged call's sealed ``fx1_completion_record.v1``
        document — ``GET /harness/completions/{id}/receipt``; 404 maps to
        KeyError. Feed it to :meth:`verify_receipt` to check the seal."""
        out = self._json(
            "GET",
            f"/harness/completions/{completion_id}/receipt",
            idempotent=True,
        )
        return dict(out)

    def check_text(self, text: str) -> GateCheckResult:
        """Pre-flight text through the remote honesty gate — POSTs
        ``/harness/gate/check``; a refusal rides ``ok=False``, it never
        raises ``Fx1HonestyError``."""
        out = self._json("POST", "/harness/gate/check", {"text": text})
        return GateCheckResult(ok=out["ok"], error=out.get("error"))

    def probe_backend(
        self,
        backend: str,
        *,
        checkpoint_dir: str | Path | None = None,
        byok: dict[str, str] | None = None,
        timeout_s: float | None = None,
        prompt: str = "ping",
    ) -> ProbeResult:
        """Remote deep-health probe — POSTs ``/harness/backends/{name}/probe``
        so a BYOK probe tests the caller's own endpoint through the same
        credential plumbing as a completion."""
        out = self._json(
            "POST",
            f"/harness/backends/{backend}/probe",
            {
                "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
                "byok": byok,
                "timeout_s": timeout_s,
                "prompt": prompt,
            },
        )
        return ProbeResult(
            backend=out["backend"],
            ok=out["ok"],
            model=out.get("model"),
            latency_ms=out["latency_ms"],
            error=out.get("error"),
            error_class=out.get("error_class"),
        )

    def complete_many(
        self,
        batch: list[list[dict[str, str]]],
        *,
        backend: str = "local_fx1",
        checkpoint_dir: str | Path | None = None,
        receipt_hashes: list[str] | None = None,
        max_workers: int = 4,
        idempotency_key: str | None = None,
        timeout_s: float | None = None,
        byok: dict[str, str] | None = None,
    ) -> list[CompletionResult]:
        """Remote counterpart of ``Fx1Harness.complete_many``.

        The server parallelizes over one shared backend; the client mirrors
        the SDK's failure contract: the lowest-index failed item raises —
        ``Fx1HonestyError`` for ``honesty_refusal``, ``RuntimeError``
        carrying the error class otherwise.
        """
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        if not batch:
            return []
        out = self._json(
            "POST",
            "/harness/complete/batch",
            {
                "backend": backend,
                "batch": batch,
                "checkpoint_dir": (str(checkpoint_dir) if checkpoint_dir is not None else None),
                "receipt_hashes": receipt_hashes,
                "timeout_s": timeout_s,
                "byok": byok,
                "max_workers": max_workers,
            },
            idempotent=idempotency_key is not None,
            extra_headers={"Idempotency-Key": idempotency_key} if idempotency_key else None,
        )
        results: list[CompletionResult] = []
        for item in out["results"]:
            if not item["ok"]:
                msg = item["error"] or "unknown backend failure"
                if item.get("error_class") == "honesty_refusal":
                    raise Fx1HonestyError(msg)
                raise RuntimeError(f"{item.get('error_class')}: {msg}")
            results.append(
                CompletionResult(
                    backend=out["backend"],
                    model=out["model"],
                    content=item["content"],
                    receipt_hashes=tuple(out["receipt_hashes"]),
                    completion_id=item.get("completion_id"),
                )
            )
        return results

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        backend: str = "local_fx1",
        checkpoint_dir: str | Path | None = None,
        receipt_hashes: list[str] | None = None,
        timeout_s: float | None = None,
        byok: dict[str, str] | None = None,
    ) -> list[str]:
        """Remote counterpart of ``Fx1Harness.stream_complete``.

        Parses the SSE token frames; the server already gated the joined
        text, so a refusal surfaces as a mapped exception, never a
        truncated stream.
        """
        _, _, body = self._request(
            "POST",
            "/harness/complete/stream",
            {
                "backend": backend,
                "messages": messages,
                "checkpoint_dir": (str(checkpoint_dir) if checkpoint_dir is not None else None),
                "receipt_hashes": receipt_hashes,
                "timeout_s": timeout_s,
                "byok": byok,
            },
        )
        chunks: list[str] = []
        saw_done = False
        for line in body.decode().splitlines():
            if not line.startswith("data: "):
                continue
            frame = line[len("data: ") :].strip()
            if frame == "[DONE]":
                saw_done = True
                break
            payload = json.loads(frame)
            if payload.get("type") == "token":
                chunks.append(payload["content"])
            elif payload.get("type") == "error":
                # Terminal in-band error (keepalive mode committed the 200
                # before the gate/backend resolved) — map it through the
                # same table as HTTP error responses.
                raise self._map_error(
                    int(payload.get("status", 502)),
                    json.dumps({"detail": payload.get("detail", "")}).encode(),
                )
        if not saw_done:
            raise HarnessTransportError("stream ended without [DONE]")
        return chunks

    # ---- receipt store -------------------------------------------------------

    def receipts(self) -> tuple[ReceiptRef, ...]:
        """GET /receipts — index the server's sealed-receipt store
        (content hash → filename, sorted)."""
        out = self._json("GET", "/receipts", idempotent=True)
        assert isinstance(out, dict)
        return tuple(ReceiptRef(sha256=item["sha256"], name=item["name"]) for item in out["items"])

    def receipt(self, sha256: str) -> StoredReceipt:
        """GET /receipts/{sha256} — the sealed document verbatim plus the
        server's live re-verify verdict (``X-Fx1-Receipt-Valid``). Unknown
        hashes raise KeyError; malformed digests raise ValueError."""
        _status, headers, body = self._request("GET", f"/receipts/{sha256}", idempotent=True)
        doc = json.loads(body)
        assert isinstance(doc, dict)
        return StoredReceipt(
            sha256=sha256,
            document=doc,
            valid=headers.get("x-fx1-receipt-valid") == "true",
        )

    # ---- receipts --------------------------------------------------------

    def verify_receipt(self, receipt: dict[str, Any]) -> ReceiptVerdict:
        """Remote counterpart of ``Fx1Harness.verify_receipt``."""
        out = self._json("POST", "/receipts/verify", {"receipt": receipt}, idempotent=True)
        return ReceiptVerdict(
            valid=out["valid"],
            path=out["path"],
            schema_tag=out["schema_tag"],
            kind=out["kind"],
            verdict=out["verdict"],
            digest_convention=out["digest_convention"],
            errors=tuple(out["errors"]),
            warnings=tuple(out["warnings"]),
        )

    def verify_receipts(self, receipts: list[dict[str, Any]]) -> tuple[ReceiptVerdict, ...]:
        """One-round-trip batch verify — the remote counterpart of looping
        ``verify_receipt``; per-item verdicts, no partial-failure abort."""
        out = self._json("POST", "/receipts/verify/batch", {"receipts": receipts}, idempotent=True)
        verdicts = []
        for item in out["results"]:
            verdicts.append(
                ReceiptVerdict(
                    valid=item["valid"],
                    path=item["path"],
                    schema_tag=item["schema_tag"],
                    kind=item["kind"],
                    verdict=item["verdict"],
                    digest_convention=item["digest_convention"],
                    errors=tuple(item["errors"]),
                    warnings=tuple(item["warnings"]),
                )
            )
        return tuple(verdicts)

    # ---- health ------------------------------------------------------------

    def health(self) -> HarnessHealth:
        """Remote counterpart of ``Fx1Harness.health``."""
        out = self._json("GET", "/health", idempotent=True)
        return HarnessHealth(
            status=out["status"],
            version=out["version"],
            registered_commands=out["registered_commands"],
            backends=dict(out["backends"]),
        )

    # ---- ops --------------------------------------------------------------

    def metrics(self) -> OpsMetrics:
        """Ops counters from the remote harness — wire-only (the in-process
        SDK has no HTTP layer to meter)."""
        out = self._json("GET", "/metrics", idempotent=True)
        return OpsMetrics(
            uptime_s=out["uptime_s"],
            requests_total=out["requests_total"],
            errors_total=out["errors_total"],
            by_status=dict(out["by_status"]),
            inflight=out["inflight"],
            inflight_watermark=out["inflight_watermark"],
            max_inflight=out["max_inflight"],
            draining=bool(out.get("draining", False)),
            rate_limited_total=int(out.get("rate_limited_total", 0)),
            complete={k: dict(v) for k, v in dict(out.get("complete", {})).items()},
        )

    def metrics_text(self) -> str:
        """Prometheus text exposition of the remote harness — same
        ``/metrics`` route, content-negotiated to ``text/plain``."""
        _, _, body = self._request(
            "GET",
            "/metrics",
            idempotent=True,
            extra_headers={"Accept": "text/plain"},
        )
        return body.decode("utf-8")

    def drain(self, wait_s: float = 0.0) -> dict[str, Any]:
        """Latch the remote harness into drain mode — one-way: gated routes
        refuse new work (503), in-flight requests finish, ``/metrics`` keeps
        reporting ``inflight`` so a deploy can wait for it to hit zero before
        stopping the process. Idempotent; marks the latch idempotent=True so
        transport blips retry. ``wait_s>0`` lets the server block until the
        in-flight pool empties — ``drained`` in the payload reports whether
        it did within the window."""
        path = "/harness/drain"
        if wait_s > 0:
            path += "?" + urllib.parse.urlencode({"wait_s": wait_s})
        out = self._json("POST", path, {}, idempotent=True)
        return {
            "draining": bool(out["draining"]),
            "inflight": int(out["inflight"]),
            "drained": bool(out.get("drained", out["inflight"] == 0)),
        }

    @property
    def last_api_version(self) -> str | None:
        """Wire-contract version stamped on the last response
        (``X-Fx1-Api-Version``); ``None`` before the first call or when
        talking to a pre-versioning server."""
        return self._last_api_version

    def server_version(self) -> dict[str, Any]:
        """GET /harness/version — the server's ``{"api_version",
        "fx1_version"}`` for version negotiation before sending work."""
        out = self._json("GET", "/harness/version", idempotent=True)
        return dict(out)

    def check_compat(self, *, strict: bool = True) -> dict[str, Any]:
        """Wire-contract negotiation: fetch ``/harness/version`` and compare
        the server's ``api_version`` against ``EXPECTED_API_VERSION`` — the
        contract this client was built to speak. Returns the report dict
        (``compatible``, both api/fx1 versions); ``strict`` (the default)
        raises ``HarnessCompatError`` on any mismatch, including a peer old
        enough to have no version route at all."""
        try:
            out = self.server_version()
            raw = out.get("api_version")
            remote: str | None = str(raw) if raw is not None else None
            fx1_version = out.get("fx1_version")
        except KeyError:
            remote = None  # peer predates the version route entirely
            fx1_version = None
        compatible = remote == EXPECTED_API_VERSION
        report = {
            "compatible": compatible,
            "client_api_version": EXPECTED_API_VERSION,
            "server_api_version": remote,
            "server_fx1_version": fx1_version,
            "client_fx1_version": __version__,
        }
        if strict and not compatible:
            raise HarnessCompatError(
                f"harness wire contract mismatch: server api_version={remote!r}, "
                f"client expects {EXPECTED_API_VERSION!r} — upgrade the server "
                f"or pin the client",
                code="incompatible_contract",
            )
        return report

    def capabilities(self) -> dict[str, Any]:
        """GET /harness/capabilities — the server's self-describing feature
        flags and effective limits (batch caps, store bounds, rate limit).
        Clients self-configure from this instead of hardcoding limits."""
        out = self._json("GET", "/harness/capabilities", idempotent=True)
        return dict(out)

    def ready(self) -> dict[str, Any]:
        """Readiness probe: returns the payload while the server accepts new
        work; raises ``BackendNotConfiguredError`` (503) once drain is
        latched — a deploy loop polls this before cutting traffic."""
        out = self._json("GET", "/ready", idempotent=True)
        return {"ready": bool(out["ready"]), "inflight": int(out["inflight"])}
