"""Resilient client for the hosted fx1 and fx1-lite chat endpoints.

``ModelClient`` is a drop-in for the factory shape Concierge already uses:
``factory(api_key=..., model=..., api_url=...)``, then
``complete_with_tools`` / ``complete``. It adds a stable error taxonomy,
bounded retries, deadlines, and cancellation. The HTTP exchange runs on a
short-lived daemon thread so the caller can abandon a blocked socket;
only the caller thread updates usage and identity.

The API key is registered with :func:`fx1.interactive.redact.register_secret`.
Exception text, ``repr``, and diagnostic events never include the key or the
URL's userinfo, path, or query.
"""

from __future__ import annotations

import contextlib
import http.client
import ipaddress
import json
import math
import os
import random
import socket
import ssl
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any, cast

from fx1.honesty import public_label, scrub_model_text
from fx1.interactive.endpoint import EndpointError, normalize_endpoint
from fx1.interactive.redact import redact_text, redact_value, register_secret
from fx1.serve.backends import (
    MOONSHOT_API_URL,
    SamplingParams,
    ToolCompletion,
    truncate_at_stops,
)

_KNOWN_MODELS = ("fx1", "fx1-lite")

_ENV_API_KEY = "MOONSHOT_API_KEY"
_ENV_BASE_URL = "FX1_BASE_URL"
_ERROR_BODY_CAP = 64 * 1024
_MAX_TOOL_CALLS = 32
_POLL_S = 0.05
# A blackholed host must not hold the prompt for the whole read timeout.
_CONNECT_TIMEOUT_S = 15.0
_JOIN_S = 0.05
_REQUEST_ID_HEADERS = ("x-request-id", "request-id", "openai-request-id")
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_RETRYABLE_SERVER = frozenset({500, 502, 503, 504})
_MODEL_CODES = frozenset(
    {
        "model_not_found",
        "model_not_available",
        "invalid_model",
        "unknown_model",
    }
)
_MODEL_PHRASES = (
    "model_not_found",
    "model not found",
    "unknown model",
    "invalid model",
    "no such model",
    "model does not exist",
    "model_not_available",
    "does not exist",
)
_CONTEXT_CODES = frozenset(
    {
        "context_length_exceeded",
        "context_too_long",
        "string_above_max_length",
        "max_tokens_exceeded",
        "token_limit_exceeded",
    }
)
_CONTEXT_PHRASES = (
    "context length",
    "context_length",
    "context window",
    "context_too_long",
    "maximum context",
    "max context",
    "too many tokens",
    "token limit",
    "string_above_max_length",
    "context exceeded",
)
_QUOTA_MARKERS = ("insufficient_quota", "insufficient quota", "billing", "quota")
_FIXED_STATUS: dict[int, tuple[ErrorKind, bool]] = {}
_HINTS: dict[ErrorKind, str] = {}
_TRANSPORT_SUMMARY = {
    "timeout": "timed out",
    "network": "connection failed",
    "tls": "tls verification failed",
    "malformed": "malformed completion",
}
_NET_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionRefusedError,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
    socket.gaierror,
    http.client.RemoteDisconnected,
)


class ErrorKind(StrEnum):
    """Why a hosted-model call failed. Values are the diagnostic strings."""

    AUTH = "auth"
    PERMISSION = "permission"
    UNKNOWN_MODEL = "unknown_model"
    ROUTE_NOT_FOUND = "route_not_found"
    RATE_LIMITED = "rate_limited"
    QUOTA = "quota"
    CONTEXT_TOO_LONG = "context_too_long"
    BAD_REQUEST = "bad_request"
    SERVER = "server"
    TIMEOUT = "timeout"
    NETWORK = "network"
    TLS = "tls"
    MALFORMED = "malformed"
    CANCELLED = "cancelled"
    DEADLINE = "deadline"


_FIXED_STATUS.update(
    {
        401: (ErrorKind.AUTH, False),
        402: (ErrorKind.QUOTA, False),
        403: (ErrorKind.PERMISSION, False),
        408: (ErrorKind.TIMEOUT, True),
    }
)
_HINTS.update(
    {
        ErrorKind.AUTH: (
            "The API key was rejected; run `/keys set {model}` with a key for this model."
        ),
        ErrorKind.PERMISSION: (
            "This key is not allowed to call the model; check the key's permissions."
        ),
        ErrorKind.UNKNOWN_MODEL: (
            "The server does not recognize model {model}; confirm the model name and endpoint."
        ),
        ErrorKind.ROUTE_NOT_FOUND: (
            "The completion route looks wrong; expected a URL ending in /v1/chat/completions."
        ),
        ErrorKind.RATE_LIMITED: ("The server is rate-limiting requests; wait before trying again."),
        ErrorKind.QUOTA: (
            "The account has no remaining quota or billing is blocked; retrying will not succeed."
        ),
        ErrorKind.CONTEXT_TOO_LONG: (
            "The prompt exceeds the model context; shorten the conversation and retry."
        ),
        ErrorKind.BAD_REQUEST: (
            "The server rejected the request; check the message and tool payload."
        ),
        ErrorKind.SERVER: (
            "The model server failed this request; retry if the error is retryable."
        ),
        ErrorKind.TIMEOUT: ("The model server did not answer before the timeout; retry."),
        ErrorKind.NETWORK: ("Could not reach {host}; check the host and the network path."),
        ErrorKind.TLS: (
            "TLS verification failed for {host}; check the certificate, "
            "the system clock, and any intercepting proxy."
        ),
        ErrorKind.MALFORMED: (
            "The server response was not a valid chat completion and was discarded."
        ),
        ErrorKind.CANCELLED: ("The model call was cancelled before it finished."),
        ErrorKind.DEADLINE: ("The model call exceeded its deadline and was stopped."),
    }
)


class ModelError(RuntimeError):
    """A hosted-model call failed. ``str(error)`` is safe to show a person."""

    kind: ErrorKind
    status: int | None
    retryable: bool
    retry_after_s: float | None
    model: str
    host: str
    hint: str
    summary: str
    request_id: str | None

    def __init__(
        self,
        kind: ErrorKind,
        *,
        status: int | None,
        retryable: bool,
        retry_after_s: float | None,
        model: str,
        host: str,
        hint: str,
        summary: str,
        request_id: str | None = None,
    ) -> None:
        self.kind = kind
        self.status = status
        self.retryable = retryable
        self.retry_after_s = retry_after_s
        self.model = model
        self.host = host
        self.summary = redact_text(summary)
        self.hint = redact_text(hint)
        self.request_id = _public_id(request_id)
        super().__init__(redact_text(f"{model} @ {host}: {self.summary} — {self.hint}"))


@dataclass(frozen=True)
class ModelIdentity:
    """Who answered one successful call. ``matched`` is None when the body omits ``model``."""

    requested: str
    served: str | None
    request_id: str | None
    host: str
    system_fingerprint: str | None
    matched: bool | None
    response_id: str | None = None


def _require_nonneg(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative number")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be a non-negative number")


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff. ``jitter`` is a fraction of the capped delay (±)."""

    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    jitter: float = 0.25
    honor_retry_after_up_to_s: float = 30.0

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise ValueError("max_attempts must be an int >= 1")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be an int >= 1")
        _require_nonneg("base_delay_s", self.base_delay_s)
        _require_nonneg("max_delay_s", self.max_delay_s)
        _require_nonneg("jitter", self.jitter)
        _require_nonneg("honor_retry_after_up_to_s", self.honor_retry_after_up_to_s)


def choose_proxy(
    host: str,
    scheme: str,
    *,
    proxies: Mapping[str, str] | None = None,
    bypass: Callable[[str], bool] | None = None,
) -> tuple[str, int] | None:
    """Return ``(proxy_host, proxy_port)`` or None for a direct connection.

    An ``https`` target prefers the ``https`` proxy and falls back to the
    ``http`` proxy. ``NO_PROXY`` / *bypass* wins. Proxy userinfo is ignored;
    the returned pair is only the hop used for ``CONNECT``.
    """
    table = dict(proxies) if proxies is not None else _default_proxies()
    bypass_fn = bypass if bypass is not None else _default_bypass
    if bypass_fn(host):
        return None
    raw = _proxy_url_for(scheme, table)
    if raw is None:
        return None
    return _parse_proxy_url(raw)


def parse_retry_after(value: str | None, *, now: float) -> float | None:
    """Seconds to wait from a ``Retry-After`` delta or HTTP-date, else None."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    seconds = _retry_after_seconds(text)
    if seconds is not None:
        return seconds
    return _retry_after_http_date(text, now)


def retry_backoff_delay(policy: RetryPolicy, attempt: int, unit: float) -> float:
    """Delay after a failed attempt (1-based) before the next try.

    ``delay = min(max_delay, base * 2**(attempt-1)) * (1 + (2*unit - 1) * jitter)``,
    clamped at 0. ``unit`` is the injected random draw in ``[0, 1]``.
    """
    exponent = max(0, attempt - 1)
    capped = min(policy.max_delay_s, policy.base_delay_s * (2.0**exponent))
    shifted = capped * (1.0 + (2.0 * unit - 1.0) * policy.jitter)
    if shifted <= 0.0:
        return 0.0
    return shifted


class _Abandoned(Exception):
    """The caller abandoned this attempt; the worker must not publish state."""


class _BadPayload(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _Oversized(Exception):
    def __init__(self, status: int) -> None:
        super().__init__("oversized")
        self.status = status


@dataclass(frozen=True)
class _Classified:
    kind: ErrorKind
    retryable: bool
    summary: str
    hint: str | None
    retry_after_s: float | None


@dataclass(frozen=True)
class _Outcome:
    completion: ToolCompletion
    usage: dict[str, int] | None
    identity: ModelIdentity
    status: int


class _Box:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.abandoned = threading.Event()
        self.lock = threading.Lock()
        self.conn: http.client.HTTPConnection | None = None
        self.error: ModelError | None = None
        self.outcome: _Outcome | None = None
        self.produced = False


class ModelClient:
    """Blocking hosted-model client. One instance may be shared across threads.

    ``deadline`` is a :func:`time.monotonic` timestamp. ``cancel_event`` and
    the deadline abort the in-flight socket; the abandoned worker does not
    update :attr:`last_identity`, :attr:`last_usage`, or :attr:`total_usage`.

    Injected ``sleep``, ``monotonic``, ``rand``, and ``now`` exist so tests
    can pin backoff without waiting. ``proxies`` / ``bypass`` override
    ``urllib.request.getproxies`` and ``proxy_bypass``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "fx1",
        api_url: str | None = None,
        timeout_s: float = 90.0,
        *,
        retry: RetryPolicy = RetryPolicy(),
        max_response_bytes: int = 4 * 1024 * 1024,
        emit: Callable[[str, Mapping[str, Any]], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        rand: Callable[[], float] = random.random,
        now: Callable[[], float] = time.time,
        proxies: Callable[[], Mapping[str, str]] | None = None,
        bypass: Callable[[str], bool] | None = None,
    ) -> None:
        key = _validated_key(api_key)
        url = _canonical_url(_resolve_url(api_url))
        self._api_key = key
        self._model = model
        self._api_url = url
        self._host = _host_label(url)
        self._path = _request_path(url)
        self._timeout_s = _require_timeout(timeout_s)
        self._retry = retry
        self._max_response_bytes = _require_cap(max_response_bytes)
        self._emit_fn = emit
        self._sleep = sleep
        self._monotonic = monotonic
        self._rand = rand
        self._now = now
        self._proxies_fn = proxies
        self._bypass_fn = bypass
        self._lock = threading.Lock()
        self._last_identity: ModelIdentity | None = None
        self._last_usage: dict[str, int] | None = None
        self._total: dict[str, int] = {}

    def __repr__(self) -> str:
        return f"ModelClient(model={self._model!r}, host={self._host!r})"

    @property
    def last_identity(self) -> ModelIdentity | None:
        with self._lock:
            return self._last_identity

    @property
    def last_usage(self) -> dict[str, int] | None:
        with self._lock:
            if self._last_usage is None:
                return None
            return dict(self._last_usage)

    @property
    def total_usage(self) -> dict[str, int]:
        """Snapshot of token counters accumulated from successful calls."""
        with self._lock:
            return dict(self._total)

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        sampling: SamplingParams | None = None,
        parallel_tool_calls: bool | None = None,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> ToolCompletion:
        """One tool-capable completion. ``deadline`` is a monotonic timestamp."""
        return self._run(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            sampling=sampling,
            parallel_tool_calls=parallel_tool_calls,
            cancel_event=cancel_event,
            deadline=deadline,
            require_text=False,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> str:
        """Text completion. A null or non-string ``content`` is ``MALFORMED``."""
        completion = self._run(
            messages,
            tools=None,
            tool_choice=None,
            sampling=sampling,
            parallel_tool_calls=None,
            cancel_event=cancel_event,
            deadline=deadline,
            require_text=True,
        )
        content = completion.content
        if not isinstance(content, str):
            raise self._error(
                ErrorKind.MALFORMED, summary="malformed completion (content is not str)"
            )
        return content

    def ping(
        self,
        *,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> ModelIdentity:
        """One-token probe used by a model check. Returns the identity of the answer."""
        self.complete(
            [{"role": "user", "content": "ping"}],
            sampling=SamplingParams(max_tokens=1),
            cancel_event=cancel_event,
            deadline=deadline,
        )
        identity = self.last_identity
        if identity is None:
            raise self._error(
                ErrorKind.MALFORMED, summary="malformed completion (missing identity)"
            )
        return identity

    def probe(
        self,
        *,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> ModelIdentity:
        """Models-route check, then a one-token completion.

        HTTP 404 and 405 on the models route are tolerated (the route may
        be absent). 401 and 403 stop before any completion. A rate limit
        on the completion is retried within the configured policy.
        """
        self._probe_models(cancel_event, deadline)
        return self.ping(cancel_event=cancel_event, deadline=deadline)

    def _probe_models(
        self,
        cancel_event: threading.Event | None,
        deadline: float | None,
    ) -> None:
        self._guard(cancel_event, deadline)
        try:
            status, headers, raw = self._get_models(cancel_event, deadline)
        except Exception as exc:
            raise self._coerce(exc) from None
        self._guard(cancel_event, deadline)
        if _models_status_ok(status):
            return
        raise self._http_error(status, headers, raw)

    def _get_models(
        self,
        cancel_event: threading.Event | None,
        deadline: float | None,
    ) -> tuple[int, dict[str, str], bytes]:
        box = _Box()
        holder: list[tuple[int, dict[str, str], bytes]] = []
        thread = threading.Thread(
            target=lambda: self._read_models_into(box, holder),
            name="fx1-model-http",
            daemon=True,
        )
        thread.start()
        try:
            self._wait_until_done(box, cancel_event, deadline)
        finally:
            if not box.done.is_set():
                self._abort(box)
            thread.join(timeout=_JOIN_S)
        if box.error is not None:
            raise box.error
        if not holder:
            raise self._error(ErrorKind.MALFORMED, summary="malformed completion (empty attempt)")
        return holder[0]

    def _read_models_into(
        self,
        box: _Box,
        holder: list[tuple[int, dict[str, str], bytes]],
    ) -> None:
        try:
            holder.append(self._read_models(box))
            box.produced = True
        except Exception as exc:
            box.error = self._coerce(exc)
        finally:
            box.done.set()

    def _read_models(self, box: _Box) -> tuple[int, dict[str, str], bytes]:
        conn = self._open_connection(self._timeout_s)
        try:
            _connect_published(conn, box)
            self._attach(box, conn)
            self._ensure_live(box)
            return _send_get(
                conn, _models_path(self._path), self._api_key, self._max_response_bytes
            )
        finally:
            _shutdown_conn(conn)

    def _run(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        sampling: SamplingParams | None,
        parallel_tool_calls: bool | None,
        cancel_event: threading.Event | None,
        deadline: float | None,
        require_text: bool,
    ) -> ToolCompletion:
        resolved = sampling if sampling is not None else SamplingParams()
        for attempt in range(1, self._retry.max_attempts + 1):
            try:
                return self._try_once(
                    messages,
                    tools,
                    tool_choice,
                    resolved,
                    parallel_tool_calls,
                    cancel_event,
                    deadline,
                    require_text,
                )
            except ModelError as exc:
                self._after_failure(exc, attempt, cancel_event, deadline)
        raise RuntimeError("model client retry loop ended without a result")

    def _try_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        sampling: SamplingParams,
        parallel_tool_calls: bool | None,
        cancel_event: threading.Event | None,
        deadline: float | None,
        require_text: bool,
    ) -> ToolCompletion:
        self._guard(cancel_event, deadline)
        self._emit_request(messages, tools)
        started = self._monotonic()
        outcome = self._one_attempt(
            messages,
            tools,
            tool_choice,
            sampling,
            parallel_tool_calls,
            cancel_event,
            deadline,
            require_text,
        )
        self._commit(outcome)
        self._emit_response(outcome, started)
        return outcome.completion

    def _after_failure(
        self,
        exc: ModelError,
        attempt: int,
        cancel_event: threading.Event | None,
        deadline: float | None,
    ) -> None:
        self._emit_error(exc)
        if not self._will_retry(exc, attempt):
            raise exc
        try:
            delay = self._delay_for(exc, attempt)
            self._ensure_room(delay, deadline)
            self._emit_retry(exc, attempt, delay)
            self._cancellable_sleep(delay, cancel_event, deadline)
        except ModelError as stop:
            self._emit_error(stop)
            raise

    def _one_attempt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        sampling: SamplingParams,
        parallel_tool_calls: bool | None,
        cancel_event: threading.Event | None,
        deadline: float | None,
        require_text: bool,
    ) -> _Outcome:
        box = _Box()
        body = _build_body(self._model, messages, sampling, tools, tool_choice, parallel_tool_calls)

        def worker() -> None:
            try:
                box.outcome = self._exchange(body, sampling, require_text, box)
                box.produced = True
            except Exception as exc:
                box.error = self._coerce(exc)
            finally:
                box.done.set()

        thread = threading.Thread(target=worker, name="fx1-model-http", daemon=True)
        thread.start()
        try:
            self._wait_until_done(box, cancel_event, deadline)
        finally:
            if not box.done.is_set():
                self._abort(box)
            thread.join(timeout=_JOIN_S)
        if box.error is not None:
            raise box.error
        if box.outcome is None:
            raise self._error(ErrorKind.MALFORMED, summary="malformed completion (empty attempt)")
        return box.outcome

    def _exchange(
        self,
        body: bytes,
        sampling: SamplingParams,
        require_text: bool,
        box: _Box,
    ) -> _Outcome:
        conn = self._open_connection(self._timeout_s)
        try:
            _connect_published(conn, box)
            self._attach(box, conn)
            self._ensure_live(box)
            status, headers, raw = _send_and_read(
                conn,
                self._path,
                body,
                self._api_key,
                self._max_response_bytes,
            )
            self._ensure_live(box)
            return self._decode(status, headers, raw, sampling, require_text)
        finally:
            _shutdown_conn(conn)

    def _decode(
        self,
        status: int,
        headers: Mapping[str, str],
        raw: bytes,
        sampling: SamplingParams,
        require_text: bool,
    ) -> _Outcome:
        if status < 200 or status >= 300:
            raise self._http_error(status, headers, raw)
        return self._success(status, headers, raw, sampling, require_text)

    def _success(
        self,
        status: int,
        headers: Mapping[str, str],
        raw: bytes,
        sampling: SamplingParams,
        require_text: bool,
    ) -> _Outcome:
        framing = _body_framing_problem(headers)
        if framing is not None:
            raise self._malformed(status, framing)
        try:
            payload = _load_json(raw)
        except (json.JSONDecodeError, ValueError, RecursionError, OverflowError):
            raise self._malformed(status, "not json") from None
        try:
            completion = _validate_completion(payload)
        except _BadPayload as exc:
            raise self._malformed(status, exc.reason) from None
        completion = _with_stops(completion, sampling)
        if require_text and not isinstance(completion.content, str):
            raise self._malformed(status, "content is not str")
        return _Outcome(
            completion=completion,
            usage=_usage_ints(payload),
            identity=_identity_from(self._model, self._host, payload, headers),
            status=status,
        )

    def _http_error(self, status: int, headers: Mapping[str, str], raw: bytes) -> ModelError:
        code, err_type, message = _error_fields(raw.decode("utf-8", errors="replace"))
        retry_after = parse_retry_after(headers.get("retry-after"), now=self._now())
        classified = _classify_http(
            status,
            code,
            err_type,
            message,
            retry_after_s=retry_after,
            honor_up_to_s=self._retry.honor_retry_after_up_to_s,
        )
        return self._error(
            classified.kind,
            status=status,
            retryable=classified.retryable,
            retry_after_s=classified.retry_after_s,
            summary=classified.summary,
            hint=classified.hint,
            request_id=_header_request_id(headers),
        )

    def _open_connection(self, timeout: float) -> http.client.HTTPConnection:
        parsed = urllib.parse.urlparse(self._api_url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        proxy = choose_proxy(
            host,
            parsed.scheme,
            proxies=self._proxy_map(),
            bypass=self._bypass_fn,
        )
        if parsed.scheme == "https":
            return _https_connection(host, port, timeout, proxy)
        return _http_connection(host, port, timeout, proxy)

    def _proxy_map(self) -> Mapping[str, str]:
        if self._proxies_fn is None:
            return _default_proxies()
        return self._proxies_fn()

    def _attach(self, box: _Box, conn: http.client.HTTPConnection) -> None:
        with box.lock:
            box.conn = conn
            abandoned = box.abandoned.is_set()
        if abandoned:
            _shutdown_conn(conn)
            raise _Abandoned()

    def _ensure_live(self, box: _Box) -> None:
        if box.abandoned.is_set():
            raise _Abandoned()

    def _wait_until_done(
        self,
        box: _Box,
        cancel_event: threading.Event | None,
        deadline: float | None,
    ) -> None:
        while not box.done.is_set():
            self._abort_if_stopped(box, cancel_event, deadline)
            box.done.wait(self._poll_timeout(cancel_event, deadline))
        if box.produced and box.error is None:
            return
        self._abort_if_stopped(box, cancel_event, deadline)

    def _abort_if_stopped(
        self,
        box: _Box,
        cancel_event: threading.Event | None,
        deadline: float | None,
    ) -> None:
        kind = _stop_kind(cancel_event, deadline, self._monotonic)
        if kind is None:
            return
        self._abort(box)
        raise self._stop_error(kind)

    def _abort(self, box: _Box) -> None:
        with box.lock:
            box.abandoned.set()
            conn = box.conn
        if conn is not None:
            _shutdown_conn(conn)

    def _poll_timeout(
        self,
        cancel_event: threading.Event | None,
        deadline: float | None,
    ) -> float | None:
        if cancel_event is None and deadline is None:
            return None
        if deadline is None:
            return _POLL_S
        remaining = deadline - self._monotonic()
        return min(_POLL_S, max(0.0, remaining))

    def _guard(self, cancel_event: threading.Event | None, deadline: float | None) -> None:
        kind = _stop_kind(cancel_event, deadline, self._monotonic)
        if kind is not None:
            raise self._stop_error(kind)

    def _will_retry(self, exc: ModelError, attempt: int) -> bool:
        if exc.kind in (ErrorKind.CANCELLED, ErrorKind.DEADLINE):
            return False
        if not exc.retryable or attempt >= self._retry.max_attempts:
            return False
        wait = exc.retry_after_s
        return not (wait is not None and wait > self._retry.honor_retry_after_up_to_s)

    def _delay_for(self, exc: ModelError, attempt: int) -> float:
        if exc.retry_after_s is not None:
            return max(0.0, exc.retry_after_s)
        return retry_backoff_delay(self._retry, attempt, self._rand())

    def _ensure_room(self, delay: float, deadline: float | None) -> None:
        if deadline is None:
            return
        if self._monotonic() + delay > deadline:
            raise self._stop_error(ErrorKind.DEADLINE)

    def _cancellable_sleep(
        self,
        delay: float,
        cancel_event: threading.Event | None,
        deadline: float | None,
    ) -> None:
        self._guard(cancel_event, deadline)
        if delay <= 0:
            return
        if cancel_event is None and deadline is None:
            self._sleep(delay)
            return
        self._sleep_slices(delay, cancel_event, deadline)

    def _sleep_slices(
        self,
        delay: float,
        cancel_event: threading.Event | None,
        deadline: float | None,
    ) -> None:
        remaining = delay
        while remaining > 0:
            self._guard(cancel_event, deadline)
            step = _sleep_step(remaining, deadline, self._monotonic)
            if step <= 0:
                raise self._stop_error(ErrorKind.DEADLINE)
            self._sleep(step)
            remaining -= step
        self._guard(cancel_event, deadline)

    def _commit(self, outcome: _Outcome) -> None:
        with self._lock:
            self._last_identity = outcome.identity
            if outcome.usage is None:
                self._last_usage = None
                return
            self._last_usage = dict(outcome.usage)
            for key, value in outcome.usage.items():
                self._total[key] = self._total.get(key, 0) + value

    def _emit_request(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> None:
        n_tools = 0 if tools is None else len(tools)
        self._emit(
            "model.request",
            {
                "model": self._model,
                "host": self._host,
                "n_messages": len(messages),
                "n_tools": n_tools,
            },
        )

    def _emit_response(self, outcome: _Outcome, started: float) -> None:
        identity = outcome.identity
        usage = dict(outcome.usage) if outcome.usage is not None else None
        self._emit(
            "model.response",
            {
                "status": outcome.status,
                "latency_ms": round((self._monotonic() - started) * 1000.0, 3),
                "usage": usage,
                "served_model": identity.served,
                "request_id": identity.request_id,
            },
        )
        if identity.matched is False:
            self._emit(
                "model.identity_mismatch",
                {
                    "requested": identity.requested,
                    "served": identity.served,
                    "host": identity.host,
                    "request_id": identity.request_id,
                },
            )

    def _emit_error(self, exc: ModelError) -> None:
        self._emit("model.error", {"kind": exc.kind.value, "status": exc.status})

    def _emit_retry(self, exc: ModelError, attempt: int, delay: float) -> None:
        self._emit(
            "model.retry",
            {
                "attempt": attempt,
                "delay_s": delay,
                "kind": exc.kind.value,
                "status": exc.status,
            },
        )

    def _emit(self, event: str, fields: Mapping[str, Any]) -> None:
        if self._emit_fn is None:
            return
        scrubbed = cast(Mapping[str, Any], redact_value(dict(fields)))
        self._emit_fn(event, scrubbed)

    def _coerce(self, exc: BaseException) -> ModelError:
        if isinstance(exc, ModelError):
            return exc
        if isinstance(exc, _Abandoned):
            return self._stop_error(ErrorKind.CANCELLED)
        if isinstance(exc, _Oversized):
            return self._error(
                ErrorKind.MALFORMED,
                status=exc.status,
                summary=f"response exceeded {self._max_response_bytes} bytes",
            )
        if isinstance(exc, _BadPayload):
            return self._malformed(None, exc.reason)
        kind = _transport_kind(exc)
        return self._error(
            kind,
            retryable=kind in (ErrorKind.TIMEOUT, ErrorKind.NETWORK),
            summary=_transport_summary(kind, exc),
        )

    def _malformed(self, status: int | None, reason: str) -> ModelError:
        return self._error(
            ErrorKind.MALFORMED,
            status=status,
            summary=f"malformed completion ({reason})",
        )

    def _stop_error(self, kind: ErrorKind) -> ModelError:
        summary = "cancelled" if kind is ErrorKind.CANCELLED else "deadline exceeded"
        return self._error(kind, summary=summary)

    def _error(
        self,
        kind: ErrorKind,
        *,
        status: int | None = None,
        retryable: bool = False,
        retry_after_s: float | None = None,
        summary: str,
        hint: str | None = None,
        request_id: str | None = None,
    ) -> ModelError:
        chosen = hint if hint is not None else _hint_for(kind, self._model, self._host)
        return ModelError(
            kind,
            status=status,
            retryable=retryable,
            retry_after_s=retry_after_s,
            model=self._model,
            host=self._host,
            hint=chosen,
            summary=summary,
            request_id=request_id,
        )


def _validated_key(api_key: str | None) -> str:
    key = _resolve_key(api_key)
    if not key:
        raise RuntimeError("MOONSHOT_API_KEY is not set; fx-1 never hardcodes credentials")
    if _key_is_unsafe(key):
        raise ValueError("API key contains whitespace, control, or non-latin1 characters")
    register_secret(key)
    return key


def _key_is_unsafe(key: str) -> bool:
    return any(_key_char_unsafe(ch) for ch in key)


def _key_char_unsafe(ch: str) -> bool:
    return ch.isspace() or unicodedata.category(ch) in {"Cc", "Cf"} or ord(ch) > 255


def _canonical_url(url: str) -> str:
    try:
        spec = normalize_endpoint(url)
    except EndpointError as exc:
        raise RuntimeError(f"model endpoint URL {exc}") from None
    return spec.completion_url


def _resolve_key(api_key: str | None) -> str:
    raw = api_key if api_key else os.environ.get(_ENV_API_KEY, "")
    if raw is None:
        return ""
    return raw.strip()


def _resolve_url(api_url: str | None) -> str:
    if api_url:
        return api_url
    env = os.environ.get(_ENV_BASE_URL, "")
    if env:
        return env
    return MOONSHOT_API_URL


def _require_timeout(timeout_s: float) -> float:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
    value = float(timeout_s)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
    return value


def _require_cap(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_response_bytes must be a positive int")
    return value


def _host_label(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "unknown"
    if parsed.port is None:
        return host
    if ":" in host:
        return f"[{host}]:{parsed.port}"
    return f"{host}:{parsed.port}"


def _request_path(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    if path:
        return path
    return "/"


def _models_path(path: str) -> str:
    suffix = "/chat/completions"
    if path.endswith(suffix):
        base = path[: -len(suffix)]
        return f"{base}/models"
    return "/v1/models"


def _models_status_ok(status: int) -> bool:
    return status in (404, 405) or 200 <= status < 300


def _default_proxies() -> dict[str, str]:
    return urllib.request.getproxies()


def _default_bypass(host: str) -> bool:
    # Plaintext loopback is the only HTTP destination that may carry a
    # bearer token. An ambient proxy would see that request, so skip it.
    if _literal_loopback(host):
        return True
    return bool(urllib.request.proxy_bypass(host))


def _literal_loopback(host: str) -> bool:
    if not host or "%" in host or host.endswith("."):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(address.is_loopback)


def _proxy_url_for(scheme: str, proxies: Mapping[str, str]) -> str | None:
    direct = proxies.get(scheme)
    if direct:
        return direct
    if scheme == "https":
        fallback = proxies.get("http")
        if fallback:
            return fallback
    return None


def _parse_proxy_url(url: str) -> tuple[str, int] | None:
    try:
        parsed = urllib.parse.urlparse(url)
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return parsed.hostname, port


def _https_connection(
    host: str,
    port: int,
    timeout: float,
    proxy: tuple[str, int] | None,
) -> http.client.HTTPConnection:
    context = ssl.create_default_context()
    if proxy is None:
        return http.client.HTTPSConnection(host, port, timeout=timeout, context=context)
    conn = http.client.HTTPSConnection(proxy[0], proxy[1], timeout=timeout, context=context)
    conn.set_tunnel(host, port)
    return conn


def _http_connection(
    host: str,
    port: int,
    timeout: float,
    proxy: tuple[str, int] | None,
) -> http.client.HTTPConnection:
    if proxy is None:
        return http.client.HTTPConnection(host, port, timeout=timeout)
    conn = http.client.HTTPConnection(proxy[0], proxy[1], timeout=timeout)
    conn.set_tunnel(host, port)
    return conn


def _retry_after_seconds(text: str) -> float | None:
    try:
        seconds = float(text)
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _retry_after_http_date(text: str, now: float) -> float | None:
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError, IndexError):
        return None
    aware = _as_utc(parsed)
    if aware is None:
        return None
    return max(0.0, aware.timestamp() - now)


def _as_utc(parsed: datetime | None) -> datetime | None:
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _fmt_seconds(seconds: float) -> str:
    rounded = round(float(seconds), 3)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.3f}".rstrip("0")


def _hint_for(kind: ErrorKind, model: str, host: str) -> str:
    return _HINTS[kind].format(model=model, host=host)


def _over_cap_hint(wait: float, cap: float) -> str:
    return (
        f"The server asked us to wait {_fmt_seconds(wait)}s, "
        f"past the retry cap of {_fmt_seconds(cap)}s; try again later."
    )


def _classify_http(
    status: int,
    code: str,
    err_type: str,
    message: str,
    *,
    retry_after_s: float | None,
    honor_up_to_s: float,
) -> _Classified:
    kind, retryable = _kind_for_status(status, code, err_type, message)
    hint = _status_hint(status, kind, retry_after_s, honor_up_to_s)
    detail = _detail(code, message)
    return _Classified(
        kind=kind,
        retryable=retryable,
        summary=_http_summary(status, detail),
        hint=hint,
        retry_after_s=retry_after_s,
    )


def _kind_for_status(status: int, code: str, err_type: str, message: str) -> tuple[ErrorKind, bool]:
    fixed = _FIXED_STATUS.get(status)
    if fixed is not None:
        return fixed
    ranged = _ranged_status(status, code, err_type, message)
    if ranged is not None:
        return ranged
    return ErrorKind.BAD_REQUEST, False


def _ranged_status(
    status: int, code: str, err_type: str, message: str
) -> tuple[ErrorKind, bool] | None:
    if status in _REDIRECT_STATUSES:
        return ErrorKind.ROUTE_NOT_FOUND, False
    if 300 <= status < 400:
        return ErrorKind.BAD_REQUEST, False
    client = _client_status(status, code, err_type, message)
    if client is not None:
        return client
    if 500 <= status <= 599:
        return ErrorKind.SERVER, status in _RETRYABLE_SERVER
    return None


def _client_status(
    status: int, code: str, err_type: str, message: str
) -> tuple[ErrorKind, bool] | None:
    if status == 404:
        return _kind_404(code, err_type, message)
    if status == 429:
        return _kind_429(code, err_type, message)
    if status in (400, 413):
        return _kind_bad_request(status, code, err_type, message)
    return None


def _kind_404(code: str, err_type: str, message: str) -> tuple[ErrorKind, bool]:
    if _is_model_problem(code, err_type, message):
        return ErrorKind.UNKNOWN_MODEL, False
    return ErrorKind.ROUTE_NOT_FOUND, False


def _kind_429(code: str, err_type: str, message: str) -> tuple[ErrorKind, bool]:
    if _is_quota(code, err_type, message):
        return ErrorKind.QUOTA, False
    return ErrorKind.RATE_LIMITED, True


def _kind_bad_request(
    status: int, code: str, err_type: str, message: str
) -> tuple[ErrorKind, bool]:
    if status == 400 and _is_model_problem(code, err_type, message):
        return ErrorKind.UNKNOWN_MODEL, False
    if status == 413 or _is_context_problem(code, err_type, message):
        return ErrorKind.CONTEXT_TOO_LONG, False
    return ErrorKind.BAD_REQUEST, False


def _status_hint(
    status: int,
    kind: ErrorKind,
    retry_after_s: float | None,
    honor_up_to_s: float,
) -> str | None:
    if status in _REDIRECT_STATUSES:
        return (
            "The server sent a redirect, which is not followed; "
            "check that the completion URL ends in /v1/chat/completions."
        )
    if kind is not ErrorKind.RATE_LIMITED or retry_after_s is None:
        return None
    if retry_after_s > honor_up_to_s:
        return _over_cap_hint(retry_after_s, honor_up_to_s)
    return None


def _is_model_problem(code: str, err_type: str, message: str) -> bool:
    if code.lower() in _MODEL_CODES:
        return True
    blob = f"{err_type} {message}".lower()
    if "model" not in blob:
        return False
    return any(phrase in blob for phrase in _MODEL_PHRASES)


def _is_context_problem(code: str, err_type: str, message: str) -> bool:
    if code.lower() in _CONTEXT_CODES:
        return True
    blob = f"{code} {err_type} {message}".lower()
    return any(phrase in blob for phrase in _CONTEXT_PHRASES)


def _is_quota(code: str, err_type: str, message: str) -> bool:
    blob = f"{code} {err_type} {message}".lower()
    return any(marker in blob for marker in _QUOTA_MARKERS)


def _detail(code: str, message: str) -> str:
    raw = code.strip() or message.strip()
    if not raw:
        return ""
    return _honest_snippet(_snippet(raw))


def _honest_snippet(text: str) -> str:
    if not text or scrub_model_text(text) == text:
        return text
    return "withheld by the honesty gate"


def _snippet(text: str, limit: int = 80) -> str:
    """Redacted excerpt. Cut first: the scrubber is quadratic on long repeated runs."""
    cleaned = " ".join(redact_text(text[:512]).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _http_summary(status: int, detail: str) -> str:
    if not detail:
        return f"HTTP {status}"
    return f"HTTP {status} ({detail})"


def _error_fields(text: str) -> tuple[str, str, str]:
    try:
        payload = _load_json(text.encode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return "", "", _snippet(text)
    if not isinstance(payload, dict):
        return "", "", ""
    error = payload.get("error")
    if not isinstance(error, dict):
        return "", "", ""
    return _text_field(error, "code"), _text_field(error, "type"), _text_field(error, "message")


def _text_field(obj: Mapping[str, Any], key: str) -> str:
    value = obj.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return ""


def _load_json(raw: bytes) -> object:
    text = raw.decode("utf-8", errors="replace")
    return cast(object, json.loads(text))


def _transport_kind(exc: BaseException) -> ErrorKind:
    if isinstance(exc, ssl.SSLError):
        return ErrorKind.TLS
    if isinstance(exc, TimeoutError):
        return ErrorKind.TIMEOUT
    if isinstance(exc, _NET_ERRORS):
        return ErrorKind.NETWORK
    if isinstance(exc, (json.JSONDecodeError, UnicodeError, http.client.BadStatusLine)):
        return ErrorKind.MALFORMED
    if isinstance(exc, OSError):
        return ErrorKind.NETWORK
    if isinstance(exc, http.client.HTTPException):
        return ErrorKind.NETWORK
    return ErrorKind.MALFORMED


def _transport_summary(kind: ErrorKind, exc: BaseException) -> str:
    label = _TRANSPORT_SUMMARY.get(kind.value, "malformed completion")
    return f"{label} ({type(exc).__name__})"


def _stop_kind(
    cancel_event: threading.Event | None,
    deadline: float | None,
    monotonic: Callable[[], float],
) -> ErrorKind | None:
    if cancel_event is not None and cancel_event.is_set():
        return ErrorKind.CANCELLED
    if deadline is not None and monotonic() >= deadline:
        return ErrorKind.DEADLINE
    return None


def _sleep_step(remaining: float, deadline: float | None, monotonic: Callable[[], float]) -> float:
    step = min(_POLL_S, remaining)
    if deadline is None:
        return step
    until_deadline = deadline - monotonic()
    if until_deadline <= 0:
        return 0.0
    return min(step, until_deadline)


def _build_body(
    model: str,
    messages: list[dict[str, Any]],
    sampling: SamplingParams,
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
    parallel_tool_calls: bool | None,
) -> bytes:
    payload: dict[str, Any] = {"model": model, "messages": messages, **sampling.body_fields()}
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = parallel_tool_calls
    return json.dumps(payload).encode("utf-8")


def _send_and_read(
    conn: http.client.HTTPConnection,
    path: str,
    body: bytes,
    api_key: str,
    max_response_bytes: int,
) -> tuple[int, dict[str, str], bytes]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    conn.request("POST", path, body=body, headers=headers)
    response = conn.getresponse()
    try:
        status = int(response.status)
        header_map = _header_map(response)
        raw = _read_response(response, status, max_response_bytes)
    finally:
        response.close()
    return status, header_map, raw


def _send_get(
    conn: http.client.HTTPConnection,
    path: str,
    api_key: str,
    max_response_bytes: int,
) -> tuple[int, dict[str, str], bytes]:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    conn.request("GET", path, headers=headers)
    response = conn.getresponse()
    try:
        status = int(response.status)
        header_map = _header_map(response)
        raw = _read_response(response, status, max_response_bytes)
    finally:
        response.close()
    return status, header_map, raw


def _body_framing_problem(headers: Mapping[str, str]) -> str | None:
    encoding = _encoding_problem(headers.get("content-encoding", ""))
    if encoding is not None:
        return encoding
    return _type_problem(headers.get("content-type", ""))


def _encoding_problem(raw: str) -> str | None:
    text = raw.strip().lower()
    if text in {"", "identity"}:
        return None
    return "unsupported content-encoding"


def _type_problem(raw: str) -> str | None:
    text = raw.strip().lower()
    if not text:
        return None
    media = text.split(";", 1)[0].strip()
    if media == "application/json" or media.endswith("+json"):
        return None
    return "content-type is not json"


def _header_map(response: http.client.HTTPResponse) -> dict[str, str]:
    found: dict[str, str] = {}
    for key, value in response.getheaders():
        lowered = key.lower()
        if lowered not in found:
            found[lowered] = value
    return found


def _read_response(
    response: http.client.HTTPResponse, status: int, max_response_bytes: int
) -> bytes:
    success = 200 <= status < 300
    limit = max_response_bytes if success else min(max_response_bytes, _ERROR_BODY_CAP)
    data = _read_bounded(response, limit, strict=success)
    if success and len(data) > max_response_bytes:
        raise _Oversized(status)
    return data


def _read_bounded(response: http.client.HTTPResponse, limit: int, *, strict: bool) -> bytes:
    budget = limit + 1 if strict else limit
    chunks: list[bytes] = []
    while budget > 0:
        chunk = response.read(min(16384, budget))
        if not chunk:
            break
        chunks.append(chunk)
        budget -= len(chunk)
    return b"".join(chunks)


def _validate_completion(payload: object) -> ToolCompletion:
    choice, message = _choice_message(payload)
    finish = choice.get("finish_reason")
    return ToolCompletion(
        content=_content(message),
        tool_calls=_tool_calls(message),
        finish_reason=finish if isinstance(finish, str) else None,
        logprobs=_logprobs(choice),
    )


def _choice_message(payload: object) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, dict):
        raise _BadPayload("response is not an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _BadPayload("missing choices[0].message")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise _BadPayload("missing choices[0].message")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise _BadPayload("missing choices[0].message")
    return choice, message


def _content(message: Mapping[str, Any]) -> str | None:
    content = message.get("content")
    if content is None:
        return None
    if isinstance(content, str) and _encodes_utf8(content):
        return content
    raise _BadPayload("content is not str|null")


def _encodes_utf8(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _logprobs(choice: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = choice.get("logprobs")
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    raise _BadPayload("logprobs is not object|null")


def _tool_calls(message: Mapping[str, Any]) -> tuple[dict[str, Any], ...] | None:
    raw = message.get("tool_calls")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise _BadPayload("tool_calls is not a list")
    if len(raw) > _MAX_TOOL_CALLS:
        raise _BadPayload(f"tool_calls exceeds {_MAX_TOOL_CALLS}")
    return tuple(_one_tool_call(item, index) for index, item in enumerate(raw))


def _one_tool_call(raw: object, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise _BadPayload(f"tool_calls[{index}] is not an object")
    function = raw.get("function")
    arguments = function.get("arguments") if isinstance(function, dict) else None
    if _tool_call_ok(raw, function, arguments):
        return raw
    raise _BadPayload(f"tool_calls[{index}] is malformed")


def _tool_call_ok(raw: Mapping[str, Any], function: object, arguments: object) -> bool:
    ident = raw.get("id")
    if not isinstance(ident, str) or raw.get("type") != "function":
        return False
    if not isinstance(function, dict):
        return False
    name = function.get("name")
    if not isinstance(name, str) or not isinstance(arguments, str):
        return False
    return _encodes_utf8(ident) and _encodes_utf8(name) and _encodes_utf8(arguments)


def _with_stops(completion: ToolCompletion, sampling: SamplingParams) -> ToolCompletion:
    content = completion.content
    if content is None or not sampling.stop:
        return completion
    trimmed = truncate_at_stops(content, sampling.stop)
    if trimmed == content:
        return completion
    return ToolCompletion(
        content=trimmed,
        tool_calls=completion.tool_calls,
        finish_reason=completion.finish_reason,
        logprobs=completion.logprobs,
    )


def _usage_ints(payload: object) -> dict[str, int] | None:
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    ints = {str(key): value for key, value in usage.items() if _is_token_count(value)}
    if not ints:
        return None
    return ints


def _is_token_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _identity_from(
    requested: str,
    host: str,
    payload: object,
    headers: Mapping[str, str],
) -> ModelIdentity:
    served = _clean_label(_served_model(payload))
    matched = None if served is None else _models_match(requested, served)
    return ModelIdentity(
        requested=requested,
        served=served,
        request_id=_clean_label(_public_id(_header_request_id(headers))),
        host=host,
        system_fingerprint=_clean_label(_fingerprint_field(payload)),
        matched=matched,
        response_id=_clean_label(_response_id(payload)),
    )


def _clean_label(value: str | None) -> str | None:
    if not value:
        return value
    return public_label(value)


def _models_match(requested: str, served: str) -> bool:
    """Exact id, or a YYYYMMDD deployment suffix of that same id.

    ``fx1-lite`` starts with ``fx1-`` but is a different model, so the
    suffix must be a date and must not name another configured model.
    """
    if served == requested:
        return True
    prefix = f"{requested}-"
    if not served.startswith(prefix):
        return False
    if _names_other_model(requested, served):
        return False
    return _is_date_suffix(served[len(prefix) :])


def _names_other_model(requested: str, served: str) -> bool:
    for other in _KNOWN_MODELS:
        if other == requested or requested.startswith(f"{other}-"):
            continue
        if served == other or served.startswith(f"{other}-"):
            return True
    return False


def _is_date_suffix(suffix: str) -> bool:
    return len(suffix) == 8 and suffix.isdigit()


def _drop_secret(value: str) -> str | None:
    """Drop a provider field when it is itself a credential."""
    if redact_text(value) != value:
        return None
    return value


def _served_model(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    if isinstance(model, str) and model:
        return _drop_secret(model)
    return None


def _fingerprint_field(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("system_fingerprint")
    if isinstance(value, str) and value:
        return _drop_secret(value)
    return None


_ID_LIMIT = 128


def _public_id(value: object) -> str | None:
    """A provider id safe to store: bounded, printable ASCII, not a secret."""
    text = _id_text(value)
    if text is None or redact_text(text) != text:
        return None
    return text


def _id_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > _ID_LIMIT:
        return None
    if any(_id_char_bad(ch) for ch in text):
        return None
    return text


def _id_char_bad(ch: str) -> bool:
    return ch.isspace() or unicodedata.category(ch) in {"Cc", "Cf"} or ord(ch) > 127


def _response_id(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    return _public_id(payload.get("id"))


def _header_request_id(headers: Mapping[str, str]) -> str | None:
    for name in _REQUEST_ID_HEADERS:
        value = headers.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _connect_published(conn: http.client.HTTPConnection, box: _Box) -> None:
    """Connect *conn* with its socket visible so cancel can close it."""
    creator = getattr(conn, "_create_connection", None)
    if not callable(creator):
        conn.connect()
        return
    setattr(conn, "_create_connection", _dial_factory(conn, box))  # noqa: B010
    try:
        conn.connect()
    finally:
        setattr(conn, "_create_connection", creator)  # noqa: B010


def _dial_factory(conn: http.client.HTTPConnection, box: _Box) -> Callable[..., socket.socket]:
    def dial(
        address: tuple[Any, ...],
        timeout: float | None = None,
        source_address: tuple[Any, ...] | None = None,
        *_args: object,
        **_kwargs: object,
    ) -> socket.socket:
        connect = timeout if timeout is None else min(timeout, _CONNECT_TIMEOUT_S)
        sock = _dial(address, connect, source_address, conn, box)
        if timeout is not None:
            sock.settimeout(timeout)
        return sock

    return dial


_STOCK_CONNECT = socket.create_connection


def _dial(
    address: tuple[Any, ...],
    timeout: float | None,
    source_address: tuple[Any, ...] | None,
    conn: http.client.HTTPConnection,
    box: _Box,
) -> socket.socket:
    if socket.create_connection is not _STOCK_CONNECT:
        return _delegated_connect(address, timeout, source_address, conn, box)
    host = ""
    port = 0
    if len(address) >= 2:
        host = str(address[0] or "")
        port = int(address[1])
    infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    return _dial_infos(list(infos), timeout, source_address, conn, box)


def _delegated_connect(
    address: tuple[Any, ...],
    timeout: float | None,
    source_address: tuple[Any, ...] | None,
    conn: http.client.HTTPConnection,
    box: _Box,
) -> socket.socket:
    sock = socket.create_connection(address, timeout, source_address)
    _publish_socket(conn, box, sock)
    return sock


def _dial_infos(
    infos: list[Any],
    timeout: float | None,
    source_address: tuple[Any, ...] | None,
    conn: http.client.HTTPConnection,
    box: _Box,
) -> socket.socket:
    last = _dial_each(infos, timeout, source_address, conn, box)
    if isinstance(last, socket.socket):
        return last
    if last is not None:
        raise last
    raise OSError("model endpoint address did not resolve")


def _dial_each(
    infos: list[Any],
    timeout: float | None,
    source_address: tuple[Any, ...] | None,
    conn: http.client.HTTPConnection,
    box: _Box,
) -> socket.socket | OSError | None:
    last: OSError | None = None
    for info in infos:
        try:
            return _dial_one(info, timeout, source_address, conn, box)
        except _Abandoned:
            raise
        except OSError as exc:
            last = exc
    return last


def _dial_one(
    info: Any,
    timeout: float | None,
    source_address: tuple[Any, ...] | None,
    conn: http.client.HTTPConnection,
    box: _Box,
) -> socket.socket:
    sock = socket.socket(int(info[0]), info[1], info[2])
    try:
        _prepare_sock(sock, timeout, source_address)
        _publish_socket(conn, box, sock)
        sock.connect(info[4])
    except _Abandoned:
        _drop_socket(conn, sock)
        raise
    except OSError:
        _drop_socket(conn, sock)
        if box.abandoned.is_set():
            raise _Abandoned() from None
        raise
    return sock


def _prepare_sock(
    sock: socket.socket,
    timeout: float | None,
    source_address: tuple[Any, ...] | None,
) -> None:
    if timeout is not None:
        sock.settimeout(timeout)
    if source_address:
        sock.bind(source_address)


def _publish_socket(conn: http.client.HTTPConnection, box: _Box, sock: socket.socket) -> None:
    with box.lock:
        conn.sock = sock
        box.conn = conn
        abandoned = box.abandoned.is_set()
    if abandoned:
        _drop_socket(conn, sock)
        raise _Abandoned()


def _drop_socket(conn: http.client.HTTPConnection, sock: socket.socket) -> None:
    if conn.sock is sock:
        conn.sock = None
    with contextlib.suppress(OSError):
        sock.close()


def _shutdown_conn(conn: http.client.HTTPConnection) -> None:
    sock = conn.sock
    if sock is not None:
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
    with contextlib.suppress(OSError):
        conn.close()


__all__ = [
    "ErrorKind",
    "ModelClient",
    "ModelError",
    "ModelIdentity",
    "RetryPolicy",
    "choose_proxy",
    "parse_retry_after",
    "retry_backoff_delay",
]
