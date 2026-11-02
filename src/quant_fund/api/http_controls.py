"""Shared HTTP controls for the lab API and the research evidence API.

Body size, in-memory token buckets, a closed CORS default, and a 500
response that carries no exception text. Loopback peers skip the rate
limit so TestClient and local probes are not throttled.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.requests import empty_receive

MAX_BODY_BYTES = 64 * 1024
BODY_TOO_LARGE_DETAIL = "request body exceeds the 64 KiB limit"
MAX_HEADER_COUNT = 128
_BAD_HEADER_BYTES = (b"\r", b"\n", b"\x00")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})
_MAX_BUCKETS = 1024
_CORS_HEADERS = (
    "access-control-allow-origin",
    "access-control-allow-credentials",
    "access-control-allow-methods",
    "access-control-allow-headers",
    "access-control-expose-headers",
)
_DISABLED = frozenset({"", "0", "off", "false", "no"})


class RequestBodyTooLarge(Exception):
    """Raised while reading a request body past the configured cap."""


class _Bucket:
    def __init__(self, rate: float, burst: float, now: float) -> None:
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.updated = now

    def take(self, now: float, rate: float, burst: float) -> bool:
        elapsed = max(0.0, now - self.updated)
        self.updated = now
        self.rate = rate
        self.burst = burst
        self.tokens = min(burst, self.tokens + elapsed * rate)
        if self.tokens < 1.0:
            return False
        self.tokens -= 1.0
        return True


class _Gate:
    def __init__(self) -> None:
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def allow(self, identity: str, rate: float, burst: float, now: float) -> bool:
        with self._lock:
            bucket = self._buckets.get(identity)
            if bucket is None:
                if len(self._buckets) >= _MAX_BUCKETS:
                    oldest = min(self._buckets, key=lambda key: self._buckets[key].updated)
                    del self._buckets[oldest]
                bucket = _Bucket(rate, burst, now)
                self._buckets[identity] = bucket
            return bucket.take(now, rate, burst)


_GATES: dict[str, _Gate] = {}


def reset_rate_gates() -> None:
    """Drop bucket state. Tests use this so cases do not share a budget."""
    _GATES.clear()


def client_host(request: Request) -> str:
    return (request.client.host if request.client else "") or ""


def is_loopback_host(host: str) -> bool:
    return host in _LOOPBACK_HOSTS


def keys_match(provided: str, expected: str) -> bool:
    """Compare API keys by SHA-256 digest so the compare length is fixed."""
    try:
        left = hashlib.sha256(provided.encode("utf-8")).digest()
        right = hashlib.sha256(expected.encode("utf-8")).digest()
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(left, right)


def nonempty_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value


def rate_config(prefix: str) -> tuple[float, float] | None:
    """``{PREFIX}_RATE_PER_SEC`` / ``{PREFIX}_RATE_BURST``.

    ``0``, ``off``, ``false``, or ``no`` disables the limit. A value that
    does not parse keeps the default of 10 requests/second and a burst of 30.
    """
    raw = os.environ.get(f"{prefix}_RATE_PER_SEC", "10").strip().lower()
    burst_raw = os.environ.get(f"{prefix}_RATE_BURST", "30").strip().lower()
    if raw in _DISABLED or burst_raw in _DISABLED:
        return None
    try:
        rate = float(raw)
        burst = float(burst_raw)
    except ValueError:
        return 10.0, 30.0
    if rate <= 0 or burst <= 0:
        return None
    return rate, burst


def _gate(prefix: str) -> _Gate:
    gate = _GATES.get(prefix)
    if gate is None:
        gate = _Gate()
        _GATES[prefix] = gate
    return gate


def _too_many() -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "rate limit exceeded"},
        headers={"Retry-After": "1"},
    )


def ip_rate_response(request: Request, prefix: str) -> JSONResponse | None:
    host = client_host(request)
    if is_loopback_host(host):
        return None
    config = rate_config(prefix)
    if config is None:
        return None
    rate, burst = config
    allowed = _gate(prefix).allow(f"{prefix}:ip:{host}", rate, burst, time.monotonic())
    if allowed:
        return None
    return _too_many()


def key_rate_response(request: Request, prefix: str, key: str) -> JSONResponse | None:
    """Per-key bucket. The bucket id is a digest, not the key."""
    if is_loopback_host(client_host(request)):
        return None
    config = rate_config(prefix)
    if config is None:
        return None
    rate, burst = config
    try:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        return None
    allowed = _gate(prefix).allow(f"{prefix}:key:{digest}", rate, burst, time.monotonic())
    if allowed:
        return None
    return _too_many()


def reject_bad_headers(request: Request) -> JSONResponse | None:
    """Refuse a header flood or a name/value that could split a response."""
    raw = request.scope.get("headers", ())
    if len(raw) > MAX_HEADER_COUNT:
        return JSONResponse(status_code=431, content={"detail": "too many headers"})
    for name, value in raw:
        if any(mark in name or mark in value for mark in _BAD_HEADER_BYTES):
            return JSONResponse(status_code=400, content={"detail": "invalid header"})
    return None


def reject_declared_body(request: Request, max_bytes: int) -> JSONResponse | None:
    bad = reject_bad_headers(request)
    if bad is not None:
        return bad
    declared = request.headers.get("content-length")
    if declared is None:
        return None
    try:
        size = int(declared)
    except ValueError:
        return JSONResponse(status_code=400, content={"detail": "invalid Content-Length"})
    if size < 0:
        return JSONResponse(status_code=400, content={"detail": "invalid Content-Length"})
    if size > max_bytes:
        return JSONResponse(status_code=413, content={"detail": BODY_TOO_LARGE_DETAIL})
    return None


def attach_body_limit(request: Request, max_bytes: int) -> None:
    """Count streamed body bytes and refuse once they pass ``max_bytes``."""
    received = 0
    original = request._receive

    async def limited_receive() -> Any:
        nonlocal received
        message = await original()
        if message.get("type") == "http.request":
            received += len(message.get("body", b""))
            if received > max_bytes:
                raise RequestBodyTooLarge
        return message

    request._receive = limited_receive


async def buffer_capped_body(request: Request) -> None:
    """Read the wrapped body now, then replay it for the route.

    The current receiver enforces the cap and raises when the streamed
    body passes it, including when ``Content-Length`` was smaller or absent.
    A message after the body, such as ``http.disconnect``, stays readable.
    A request built without a receive channel is left untouched.
    """
    original = request._receive
    if original is empty_receive:
        return
    chunks: list[bytes] = []
    while True:
        message = await original()
        if message.get("type") != "http.request":
            _replace_receive(request, b"".join(chunks), original, message)
            return
        body = message.get("body", b"") or b""
        if body:
            chunks.append(body)
        if not message.get("more_body", False):
            break
    _replace_receive(request, b"".join(chunks), original, None)


def _replace_receive(
    request: Request,
    payload: bytes,
    original: Any,
    pending: Any,
) -> None:
    queued: list[Any] = [{"type": "http.request", "body": payload, "more_body": False}]
    if pending is not None:
        queued.append(pending)

    async def replay() -> Any:
        if queued:
            return queued.pop(0)
        return await original()

    request._receive = replay


def apply_security_headers(response: Any, scheme: str) -> None:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'"
    )
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Cache-Control"] = "no-store"
    if scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"


def enforce_closed_cors(response: Any) -> None:
    """Drop CORS allow headers. The default policy reflects no origin."""
    for name in _CORS_HEADERS:
        if name in response.headers:
            del response.headers[name]


def install_plain_errors(app: FastAPI) -> None:
    """Turn unhandled errors into a fixed JSON body with no exception text.

    This handler runs outside user middleware, so the response carries the
    same security headers as a route response.
    """

    @app.exception_handler(Exception)
    async def plain_internal_error(request: Request, _exc: Exception) -> JSONResponse:
        response = JSONResponse(status_code=500, content={"detail": "internal error"})
        apply_security_headers(response, request.url.scheme)
        enforce_closed_cors(response)
        return response
