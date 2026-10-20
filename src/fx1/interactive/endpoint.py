"""Endpoint normalization plus failure guidance, retries, and identity probes.

:func:`normalize_endpoint` turns pasted service URLs into chat-completions
routes. Guidance helpers classify backend failures without echoing secrets.
"""

from __future__ import annotations

import math
import random
import re
import threading
import time
import unicodedata
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from fx1.serve.backends import hosted_api_url_problem

_COMPLETION_ROUTE = "/chat/completions"
_DEFAULT_PREFIX = "/v1"
_MAX_ENDPOINT_CHARS = 2048
_EMBEDDED_REJECT = ("\x00", "\n", "\r")


class EndpointError(ValueError):
    """The endpoint text could not be turned into a completion URL.

    The message is a reason only. It never includes the supplied text,
    userinfo, query, or token.
    """


@dataclass(frozen=True)
class EndpointSpec:
    """A validated completion endpoint."""

    origin: str
    api_prefix: str
    completion_url: str
    host: str
    route_added: bool

    def describe(self) -> str:
        """``POST`` plus the completion URL, for display."""
        return f"POST {self.completion_url}"


def normalize_endpoint(raw: str) -> EndpointSpec:
    """Return the canonical completion endpoint for *raw*."""
    cleaned = _clean_input(raw)
    parsed = _parse_http_url(cleaned)
    _reject_decorations(cleaned, parsed)
    host = _host_name(parsed)
    origin = _origin(parsed, host)
    prefix, route_added = _split_prefix(parsed.path)
    completion = _completion_url(origin, prefix)
    _reject_absurd_length(completion)
    _require_safe(completion)
    return EndpointSpec(
        origin=origin,
        api_prefix=prefix,
        completion_url=completion,
        host=host,
        route_added=route_added,
    )


def _clean_input(raw: str) -> str:
    if any(ch in raw for ch in _EMBEDDED_REJECT):
        raise EndpointError("endpoint must not contain newlines or NUL bytes")
    _reject_absurd_length(raw)
    cleaned = "".join(ch for ch in raw if not _is_ignorable(ch))
    if not cleaned:
        raise EndpointError("endpoint must not be empty")
    return cleaned


def _is_ignorable(ch: str) -> bool:
    return ch.isspace() or unicodedata.category(ch) in {"Cc", "Cf"}


def _reject_absurd_length(text: str) -> None:
    if len(text) > _MAX_ENDPOINT_CHARS:
        raise EndpointError("endpoint must be at most 2048 characters")


def _parse_http_url(cleaned: str) -> urllib.parse.ParseResult:
    try:
        parsed = urllib.parse.urlparse(cleaned)
    except ValueError:
        raise EndpointError("must be a well-formed http(s) URL") from None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise EndpointError("must be an http(s) URL")
    return parsed


def _reject_decorations(cleaned: str, parsed: urllib.parse.ParseResult) -> None:
    if parsed.username is not None or parsed.password is not None:
        raise EndpointError("must not embed userinfo credentials")
    if parsed.params or parsed.query or "?" in cleaned:
        raise EndpointError("must not carry params, query, or fragment")
    if parsed.fragment or "#" in cleaned:
        raise EndpointError("must not carry params, query, or fragment")


def _host_name(parsed: urllib.parse.ParseResult) -> str:
    host = parsed.hostname
    if not host:
        raise EndpointError("must name a host")
    lowered = host.lower()
    if not lowered.isascii():
        raise EndpointError("hostname must be ASCII")
    if any(label.startswith("xn--") for label in lowered.split(".")):
        raise EndpointError("punycode hostnames are not accepted")
    return lowered


def _origin(parsed: urllib.parse.ParseResult, host: str) -> str:
    try:
        port = parsed.port
    except ValueError:
        raise EndpointError("port must be 1-65535") from None
    shown = f"[{host}]" if ":" in host else host
    scheme = parsed.scheme.lower()
    if port is None:
        return f"{scheme}://{shown}"
    return f"{scheme}://{shown}:{port}"


def _split_prefix(path: str) -> tuple[str, bool]:
    collapsed = _collapse_path(path)
    if collapsed in {"", "/"}:
        return _DEFAULT_PREFIX, True
    if collapsed.endswith(_COMPLETION_ROUTE):
        return collapsed[: -len(_COMPLETION_ROUTE)], False
    return collapsed, True


def _collapse_path(path: str) -> str:
    if not path:
        return ""
    collapsed = re.sub(r"/{2,}", "/", path)
    if collapsed != "/":
        collapsed = collapsed.rstrip("/")
    return collapsed


def _completion_url(origin: str, prefix: str) -> str:
    return f"{origin}{prefix}{_COMPLETION_ROUTE}"


def _require_safe(url: str) -> None:
    problem = hosted_api_url_problem(url)
    if problem is None:
        return
    if url in problem:
        raise EndpointError("endpoint was rejected")
    raise EndpointError(problem)


_HTTP_STATUS_RE = re.compile(r"HTTP (\d{3})\b")
_URL_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s]+")
_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)
_NAMED_TOKEN_RE = re.compile(r"\b(?:sk|key)-[A-Za-z0-9_\-]+", re.IGNORECASE)
_USERINFO_RE = re.compile(r"(?P<user>[A-Za-z0-9._%+\-]+):(?P<pw>[^\s/@]+)@(?P<host>[A-Za-z0-9.-]+)")
_BLOB_RE = re.compile(r"\b[A-Za-z0-9]{32,}\b")
_URL_TRAILING = ".,);]>'\"/"

_REQUEST_REJECTED = (
    "request rejected; the model id '{model}' may be unknown to this service or the payload invalid"
)


class FailureClass(StrEnum):
    """Why a hosted endpoint call failed."""

    AUTH = "auth"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    SERVER = "server"
    TIMEOUT = "timeout"
    UNREACHABLE = "unreachable"
    MALFORMED = "malformed"
    UNCONFIGURED = "unconfigured"
    BAD_URL = "bad_url"
    CANCELLED = "cancelled"
    IDENTITY_MISMATCH = "identity_mismatch"
    IDENTITY_UNVERIFIED = "identity_unverified"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Guidance:
    """Secret-free explanation of one backend failure."""

    failure: FailureClass
    summary: str
    action: str
    retryable: bool
    http_status: int | None


_HTTP_CLASS: dict[int, FailureClass] = {
    400: FailureClass.MALFORMED,
    401: FailureClass.AUTH,
    403: FailureClass.AUTH,
    404: FailureClass.NOT_FOUND,
    408: FailureClass.RATE_LIMITED,
    422: FailureClass.MALFORMED,
    429: FailureClass.RATE_LIMITED,
}

_RETRYABLE: dict[FailureClass, bool] = {
    FailureClass.AUTH: False,
    FailureClass.NOT_FOUND: False,
    FailureClass.RATE_LIMITED: True,
    FailureClass.SERVER: True,
    FailureClass.TIMEOUT: True,
    FailureClass.UNREACHABLE: True,
    FailureClass.MALFORMED: False,
    FailureClass.UNCONFIGURED: False,
    FailureClass.BAD_URL: False,
    FailureClass.CANCELLED: False,
    FailureClass.IDENTITY_MISMATCH: False,
    FailureClass.IDENTITY_UNVERIFIED: False,
    FailureClass.UNKNOWN: False,
}

_SUMMARIES: dict[FailureClass, str] = {
    FailureClass.AUTH: "authentication failed",
    FailureClass.NOT_FOUND: "the endpoint route was not found",
    FailureClass.RATE_LIMITED: "the endpoint asked the client to retry later",
    FailureClass.SERVER: "the model server failed; the endpoint returned a server error",
    FailureClass.TIMEOUT: "the endpoint timed out",
    FailureClass.UNREACHABLE: "the endpoint is unreachable (DNS or connection)",
    FailureClass.MALFORMED: "the endpoint returned a malformed response",
    FailureClass.UNCONFIGURED: "the model endpoint is not configured",
    FailureClass.BAD_URL: "the endpoint URL is invalid",
    FailureClass.CANCELLED: "the operation was cancelled",
    FailureClass.IDENTITY_MISMATCH: "the served model id differs from the requested id",
    FailureClass.IDENTITY_UNVERIFIED: "the provider did not report a served model id",
}

_ACTIONS: dict[FailureClass, str] = {
    FailureClass.AUTH: "check the API key for {model}: /keys set {model}",
    FailureClass.NOT_FOUND: (
        "the endpoint URL must be the full chat-completions route "
        "(e.g. https://<host>/v1/chat/completions); fix it with /keys set {model}"
    ),
    FailureClass.RATE_LIMITED: "wait and retry; reduce concurrent runs",
    FailureClass.SERVER: "wait and retry",
    FailureClass.TIMEOUT: "wait and retry",
    FailureClass.UNREACHABLE: "check the network and the endpoint host",
    FailureClass.MALFORMED: "confirm the model id and the chat-completions payload",
    FailureClass.UNCONFIGURED: "/keys set {model}",
    FailureClass.BAD_URL: "/keys set {model} with an https:// URL",
    FailureClass.CANCELLED: "rerun the command",
    FailureClass.IDENTITY_MISMATCH: "check provider routing and the model deployment",
    FailureClass.IDENTITY_UNVERIFIED: "use an endpoint that reports its served model id",
    FailureClass.UNKNOWN: "retry the command or inspect the endpoint configuration",
}


def explain_backend_error(exc: BaseException, *, model: str = "fx1") -> Guidance:
    """Classify ``exc`` for ``model`` without echoing secrets from its text."""
    failure, status = _classify(exc, str(exc))
    return Guidance(
        failure=failure,
        summary=_summary(failure, exc, model, status),
        action=_action(failure, model),
        retryable=_RETRYABLE[failure],
        http_status=status,
    )


def format_guidance(guidance: Guidance) -> str:
    """One line: class, optional HTTP status, summary, and the fix."""
    label = _failure_label(guidance.failure, guidance.http_status, guidance.retryable)
    return f"model endpoint error ({label}): {guidance.summary} → fix: {guidance.action}"


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff for retryable endpoint failures.

    ``max_retries`` is the number of retries after the first attempt.
    The wait before retry ``n`` (1-based) is
    ``min(max_delay_s, base_delay_s * 2**(n - 1))`` scaled by
    ``1 + jitter * U(-1, 1)``.
    """

    max_retries: int = 2
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    jitter: float = 0.1

    def validate(self) -> None:
        """Reject negative or non-finite bounds, jitter above 1, and ``max_delay_s < base_delay_s``."""
        _require_retries(self.max_retries)
        _require_non_negative("base_delay_s", self.base_delay_s)
        _require_non_negative("max_delay_s", self.max_delay_s)
        _require_non_negative("jitter", self.jitter)
        if self.max_delay_s < self.base_delay_s:
            raise ValueError("max_delay_s must be >= base_delay_s")
        if self.jitter > 1:
            raise ValueError("jitter must be between 0 and 1 inclusive")


class OperationCancelled(RuntimeError):
    """A model attempt or retry wait was aborted by cancellation."""


def call_with_retries[T](
    fn: Callable[[], T],
    *,
    policy: RetryPolicy | None = None,
    cancel_event: threading.Event | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, Guidance, float], None] | None = None,
    rng: random.Random | None = None,
    model: str = "fx1",
) -> T:
    """Call ``fn`` until it returns, the policy is exhausted, or it must not retry.

    Retryable failures wait, then try again. Auth and other definitive
    failures re-raise the original exception. ``KeyboardInterrupt``
    propagates immediately. A set ``cancel_event`` prevents an attempt from
    starting and interrupts a retry before the next attempt.
    """
    active = RetryPolicy() if policy is None else policy
    active.validate()
    generator = random.Random() if rng is None else rng
    return _attempt_loop(
        fn,
        policy=active,
        cancel_event=cancel_event,
        sleep=sleep,
        on_retry=on_retry,
        rng=generator,
        model=model,
    )


@dataclass(frozen=True)
class IdentityReport:
    """Result of one identity probe, success or classified failure."""

    requested_model: str
    served_model: str | None
    ok: bool
    latency_ms: float
    http_status: int | None
    failure: FailureClass | None
    detail: str
    endpoint_host: str
    checked_at: str

    @property
    def identity_matches(self) -> bool | None:
        """Whether the served id is exactly the requested model id.

        ``None`` when the provider did not report a model id.
        """
        served = self.served_model
        if served is None:
            return None
        return served == self.requested_model

    def render(self) -> str:
        """Multi-line status safe to print in the chat transcript."""
        if self.ok:
            return _render_ok(self)
        return _render_failed(self)


def probe_identity(
    backend: Any,
    *,
    model: str,
    endpoint_host: str,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] | None = None,
) -> IdentityReport:
    """Call ``backend.identity_probe()`` and always return a report.

    The backend is duck-typed. A returned mapping may include ``model``,
    ``id``, and ``usage``; only ``model`` is read. Exceptions become a
    failed report via :func:`explain_backend_error` and are not re-raised.
    """
    checked_at = _checked_at(now)
    started = clock()
    try:
        payload = backend.identity_probe()
        served = _served_model(payload)
    except Exception as exc:
        return _failed_report(
            exc,
            model=model,
            endpoint_host=endpoint_host,
            latency_ms=_elapsed_ms(clock, started),
            checked_at=checked_at,
        )
    return _ok_report(
        served,
        model=model,
        endpoint_host=endpoint_host,
        latency_ms=_elapsed_ms(clock, started),
        checked_at=checked_at,
    )


def _classify(exc: BaseException, message: str) -> tuple[FailureClass, int | None]:
    if _is_cancelled(exc):
        return FailureClass.CANCELLED, None
    status = _http_status(message)
    from_http = _from_http(status)
    if from_http is not None:
        return from_http, status
    from_text = _from_text(message, exc)
    if from_text is not None:
        return from_text, None
    return FailureClass.UNKNOWN, status


def _is_cancelled(exc: BaseException) -> bool:
    return isinstance(exc, KeyboardInterrupt) or "Cancelled" in type(exc).__name__


def _http_status(message: str) -> int | None:
    match = _HTTP_STATUS_RE.search(message)
    if match is None:
        return None
    return int(match.group(1))


def _from_http(status: int | None) -> FailureClass | None:
    if status is None:
        return None
    exact = _HTTP_CLASS.get(status)
    if exact is not None:
        return exact
    if 500 <= status <= 599:
        return FailureClass.SERVER
    return None


def _from_text(message: str, exc: BaseException) -> FailureClass | None:
    if isinstance(exc, TimeoutError):
        return FailureClass.TIMEOUT
    lowered = message.lower()
    if "unreachable" in lowered:
        return FailureClass.UNREACHABLE
    if "transport fault" in lowered:
        return _transport_fault_class(lowered)
    if "not json" in lowered or "malformed" in lowered:
        return FailureClass.MALFORMED
    if _is_unconfigured(lowered):
        return FailureClass.UNCONFIGURED
    if _is_bad_url(message):
        return FailureClass.BAD_URL
    return None


def _transport_fault_class(lowered: str) -> FailureClass:
    if "timeout" in lowered or "timed out" in lowered:
        return FailureClass.TIMEOUT
    return FailureClass.UNREACHABLE


def _is_unconfigured(lowered: str) -> bool:
    return "is not set" in lowered or "no endpoint" in lowered or "not configured" in lowered


def _is_bad_url(message: str) -> bool:
    lowered = message.lower()
    if "https url" in lowered:
        return True
    return "invalid" in lowered and "url" in lowered


def _summary(failure: FailureClass, exc: BaseException, model: str, status: int | None) -> str:
    if failure is FailureClass.UNKNOWN:
        return _unknown_summary(exc)
    if failure is FailureClass.MALFORMED and status in (400, 422):
        return _REQUEST_REJECTED.replace("{model}", model)
    return _SUMMARIES[failure]


def _unknown_summary(exc: BaseException) -> str:
    detail = _redact(str(exc))
    name = type(exc).__name__
    if not detail:
        return name
    return f"{name}: {detail}"


def _redact(message: str) -> str:
    cleaned = _URL_RE.sub(_url_to_host, message)
    cleaned = _BEARER_RE.sub("", cleaned)
    cleaned = _NAMED_TOKEN_RE.sub("", cleaned)
    cleaned = _USERINFO_RE.sub(_userinfo_to_host, cleaned)
    cleaned = _BLOB_RE.sub("", cleaned)
    return " ".join(cleaned.split())


def _url_to_host(match: re.Match[str]) -> str:
    token = match.group(0)
    trimmed = token.rstrip(_URL_TRAILING)
    host = _hostname(trimmed)
    if not host:
        return ""
    return host


def _hostname(url: str) -> str:
    try:
        host = urllib.parse.urlparse(url).hostname
    except ValueError:
        return ""
    if not host or "@" in host or ":" in host:
        return ""
    return host


def _userinfo_to_host(match: re.Match[str]) -> str:
    return match.group("host")


def _action(failure: FailureClass, model: str) -> str:
    return _ACTIONS[failure].replace("{model}", model)


def _failure_label(failure: FailureClass, http_status: int | None, retryable: bool) -> str:
    name = failure.value
    if retryable:
        name = f"{name} (retryable)"
    if http_status is None:
        return name
    return f"{name}, HTTP {http_status}"


def _require_retries(max_retries: int) -> None:
    if isinstance(max_retries, bool) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")


def _require_non_negative(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite non-negative number")
    if value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")


def _attempt_loop[T](
    fn: Callable[[], T],
    *,
    policy: RetryPolicy,
    cancel_event: threading.Event | None,
    sleep: Callable[[float], None],
    on_retry: Callable[[int, Guidance, float], None] | None,
    rng: random.Random,
    model: str,
) -> T:
    attempt = 0
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise OperationCancelled(f"cancelled before attempt {attempt + 1}")
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            guidance = explain_backend_error(exc, model=model)
            if not guidance.retryable or attempt >= policy.max_retries:
                raise
            _wait_retry(
                exc,
                guidance,
                attempt=attempt,
                policy=policy,
                cancel_event=cancel_event,
                sleep=sleep,
                on_retry=on_retry,
                rng=rng,
            )
            attempt += 1


def _wait_retry(
    exc: Exception,
    guidance: Guidance,
    *,
    attempt: int,
    policy: RetryPolicy,
    cancel_event: threading.Event | None,
    sleep: Callable[[float], None],
    on_retry: Callable[[int, Guidance, float], None] | None,
    rng: random.Random,
) -> None:
    delay = _delay_for(policy, attempt, rng)
    retry_n = attempt + 1
    if on_retry is not None:
        on_retry(retry_n, guidance, delay)
    _ensure_not_cancelled(cancel_event, retry_n, exc)
    sleep(delay)
    _ensure_not_cancelled(cancel_event, retry_n, exc)


def _ensure_not_cancelled(
    cancel_event: threading.Event | None, retry_n: int, exc: BaseException
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise OperationCancelled(f"cancelled before retry {retry_n}") from exc


def _delay_for(policy: RetryPolicy, attempt: int, rng: random.Random) -> float:
    capped = min(policy.max_delay_s, policy.base_delay_s * (2.0**attempt))
    return capped * (1.0 + policy.jitter * rng.uniform(-1.0, 1.0))


def _render_ok(report: IdentityReport) -> str:
    return (
        f"{report.requested_model} @ {report.endpoint_host}: OK — "
        f"served model {report.served_model!r} in {round(report.latency_ms)} ms"
    )


def _render_failed(report: IdentityReport) -> str:
    failure = FailureClass.UNKNOWN if report.failure is None else report.failure
    label = _failure_label(failure, report.http_status, _RETRYABLE[failure])
    action = _action(failure, report.requested_model)
    return (
        f"{report.requested_model} @ {report.endpoint_host}: "
        f"FAILED ({label}) — {report.detail} → fix: {action}"
    )


def _checked_at(now: Callable[[], datetime] | None) -> str:
    moment = datetime.now(UTC) if now is None else now()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def _elapsed_ms(clock: Callable[[], float], started: float) -> float:
    return (clock() - started) * 1000.0


def _served_model(payload: object) -> str | None:
    if not isinstance(payload, dict):
        raise RuntimeError("malformed identity probe payload")
    reported = payload.get("model")
    if isinstance(reported, str) and reported:
        return reported
    return None


def _failed_report(
    exc: Exception,
    *,
    model: str,
    endpoint_host: str,
    latency_ms: float,
    checked_at: str,
) -> IdentityReport:
    guidance = explain_backend_error(exc, model=model)
    return IdentityReport(
        requested_model=model,
        served_model=None,
        ok=False,
        latency_ms=latency_ms,
        http_status=guidance.http_status,
        failure=guidance.failure,
        detail=guidance.summary,
        endpoint_host=endpoint_host,
        checked_at=checked_at,
    )


def _ok_report(
    served: str | None,
    *,
    model: str,
    endpoint_host: str,
    latency_ms: float,
    checked_at: str,
) -> IdentityReport:
    if served is None:
        detail = f"provider did not report a model id; requested {model!r} remains unverified"
        failure = FailureClass.IDENTITY_UNVERIFIED
    elif served != model:
        detail = f"requested model {model!r}; served model {served!r}"
        failure = FailureClass.IDENTITY_MISMATCH
    else:
        detail = f"served model {served!r}"
        failure = None
    return IdentityReport(
        requested_model=model,
        served_model=served,
        ok=failure is None,
        latency_ms=latency_ms,
        http_status=None,
        failure=failure,
        detail=detail,
        endpoint_host=endpoint_host,
        checked_at=checked_at,
    )
