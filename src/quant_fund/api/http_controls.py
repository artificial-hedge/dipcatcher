"""Shared HTTP controls for the lab API and the research evidence API.

Body size, token buckets, a closed CORS default, and a 500 response that
carries no exception text. Loopback peers use a separate, higher token
bucket than other peers. The default bucket is in-process. ``sqlite`` and
``file`` share one store across workers.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.requests import empty_receive

from quant_fund.api.rate_store import (
    backend_name,
    shared_take,
)
from quant_fund.api.rate_store import (
    rate_backend_env_names as _rate_backend_env_names,
)

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


_STAND_IN_KEY = "absent-api-key"
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_READ_DENIED_DETAIL = "read key cannot modify"
_FAILED_PRESENTED = b"presented-unencodable"
_FAILED_EXPECTED = b"expected-unencodable"
_JSON_MAX_DEPTH = 32


def _sha256_key(value: str, failed: bytes) -> bytes:
    try:
        return hashlib.sha256(value.encode("utf-8")).digest()
    except UnicodeEncodeError:
        return hashlib.sha256(failed).digest()


def keys_match(provided: str, expected: str) -> bool:
    """Compare API keys by SHA-256 digest so the compare length is fixed.

    An unencodable string still reaches ``compare_digest``. It does not match
    a key that encoded, and it does not return before the compare.
    """
    left = _sha256_key(provided, _FAILED_PRESENTED)
    right = _sha256_key(expected, _FAILED_EXPECTED)
    return hmac.compare_digest(left, right)


def auth_decision(provided: str | None, expected: str | None) -> str:
    """Return ``ok``, ``unauthorized``, or ``forbidden`` after one digest compare.

    A missing header and an unset server key still run ``keys_match``, so a
    401 and a 403 do the same digest work.
    """
    configured = bool(expected)
    presented = provided if provided else _STAND_IN_KEY
    reference = expected if expected else _STAND_IN_KEY
    matched = keys_match(presented, reference)
    if configured and bool(provided) and matched:
        return "ok"
    if configured:
        return "unauthorized"
    return "forbidden"


def _presented(value: str | None) -> str:
    if value:
        return value
    return _STAND_IN_KEY


def _configured_match(provided: str | None, expected: str | None) -> bool:
    """Digest both sides, then accept only a configured key that was sent."""
    matched = keys_match(_presented(provided), _presented(expected))
    if not expected:
        return False
    if not provided:
        return False
    return matched


def _slot_match(provided: str | None, current: str | None, previous: str | None) -> bool:
    """Digest the current key and the previous key before deciding.

    The previous compare always runs, including when the current key matched
    and when the previous key is unset.
    """
    current_ok = _configured_match(provided, current)
    previous_ok = _configured_match(provided, previous)
    return current_ok or previous_ok


def _any_key(*keys: str | None) -> bool:
    return any(keys)


def admin_key_matches(
    provided: str | None,
    current: str | None,
    previous: str | None,
) -> bool:
    """True when the header matches the current or previous admin key."""
    return _slot_match(provided, current, previous)


def scoped_auth_decision(
    method: str,
    provided: str | None,
    admin_key: str | None,
    read_key: str | None,
    admin_previous: str | None = None,
    read_previous: str | None = None,
) -> str:
    """Return ``ok``, ``unauthorized``, ``forbidden``, or ``read_denied``.

    The admin key authorizes every method. The read key authorizes safe
    methods only. Current and previous digests both run before the decision,
    including when a key is unset and when the current key matches.
    """
    admin_ok = _slot_match(provided, admin_key, admin_previous)
    read_ok = _slot_match(provided, read_key, read_previous)
    if admin_ok:
        return "ok"
    if read_ok and method.upper() in _SAFE_METHODS:
        return "ok"
    if read_ok:
        return "read_denied"
    if _any_key(admin_key, read_key, admin_previous, read_previous):
        return "unauthorized"
    return "forbidden"


def credential_refusal(
    request: Request,
    admin_key: str | None,
    read_key: str | None,
    unset_detail: str,
    admin_previous: str | None = None,
    read_previous: str | None = None,
) -> JSONResponse | None:
    """401, 403, or None when the request may proceed."""
    decision = scoped_auth_decision(
        request.method,
        request.headers.get("X-API-Key"),
        admin_key,
        read_key,
        admin_previous,
        read_previous,
    )
    if decision == "ok":
        return None
    if decision == "read_denied":
        return JSONResponse(status_code=403, content={"detail": _READ_DENIED_DETAIL})
    if decision != "forbidden":
        return auth_rejection("unauthorized", unset_detail=unset_detail)
    if is_loopback_host(client_host(request)):
        return None
    return auth_rejection("forbidden", unset_detail=unset_detail)


def auth_rejection(decision: str, *, unset_detail: str) -> JSONResponse:
    """401 when a key is configured. 403 when the server key is unset."""
    if decision == "forbidden":
        return JSONResponse(status_code=403, content={"detail": unset_detail})
    return JSONResponse(status_code=401, content={"detail": "invalid or missing X-API-Key"})


def serve_banner_kwargs() -> dict[str, bool]:
    """Uvicorn flags that omit the server and date banners."""
    return {"server_header": False, "date_header": False}


def _json_string_step(byte: int, escaped: bool) -> tuple[bool, bool]:
    """Advance one byte inside a JSON string. Returns (in_string, escaped)."""
    if escaped:
        return True, False
    if byte == 0x5C:
        return True, True
    if byte == 0x22:
        return False, False
    return True, False


def json_nesting_too_deep(payload: bytes, max_depth: int = _JSON_MAX_DEPTH) -> bool:
    """True when raw JSON nesting exceeds ``max_depth``.

    The scan is iterative. A document that would blow the parser's recursion
    limit is rejected before a route reads it.
    """
    depth = 0
    in_string = False
    escaped = False
    for byte in payload:
        if in_string:
            in_string, escaped = _json_string_step(byte, escaped)
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x7B, 0x5B):
            depth += 1
            if depth > max_depth:
                return True
        elif byte in (0x7D, 0x5D) and depth:
            depth -= 1
    return False


def reject_deep_json(request: Request) -> JSONResponse | None:
    payload = getattr(request.state, "bounded_body", b"")
    if isinstance(payload, bytes) and payload and json_nesting_too_deep(payload):
        return JSONResponse(status_code=400, content={"detail": "JSON document is too deep"})
    return None


def nonempty_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value


_SECRET_FILE_BYTES = 4096
_UNREADABLE_SECRET = "file-secret-unreadable-" + secrets.token_hex(16)
_KEY_FILE_SUFFIXES = ("_FILE", "_PREVIOUS_FILE", "_GRACE_UNTIL_FILE")
_KEY_SLOTS = (
    "QUANT_API_KEY",
    "QUANT_API_READ_KEY",
    "RESEARCH_API_KEY",
    "RESEARCH_API_READ_KEY",
)


def key_file_env_names() -> tuple[str, ...]:
    """Environment names that point at key material or a grace deadline."""
    return tuple(f"{slot}{suffix}" for slot in _KEY_SLOTS for suffix in _KEY_FILE_SUFFIXES)


def admin_key_configured(name: str) -> bool:
    """True when the admin env value or its ``*_FILE`` path is set.

    A read-key file does not satisfy this check. The bind guard uses it so a
    file-backed admin key can listen beyond loopback.
    """
    if nonempty_env(name):
        return True
    return bool(os.environ.get(f"{name}_FILE", "").strip())


def load_key_slot(name: str, *, now: datetime | None = None) -> tuple[str | None, str | None]:
    """Current and previous secrets for one key name.

    ``{name}_FILE`` replaces the environment value. ``{name}_PREVIOUS_FILE``
    is accepted only while ``now`` is strictly before the timestamp in
    ``{name}_GRACE_UNTIL_FILE``. A set file that cannot be read stays
    configured and matches nothing.
    """
    moment = now if now is not None else datetime.now(UTC)
    return _current_secret(name), _previous_secret(name, moment)


def _current_secret(name: str) -> str | None:
    if os.environ.get(f"{name}_FILE", "").strip():
        return _secret_from_env_path(f"{name}_FILE")
    return nonempty_env(name)


def _previous_secret(name: str, now: datetime) -> str | None:
    if not _grace_open(name, now):
        return None
    env_name = f"{name}_PREVIOUS_FILE"
    if not os.environ.get(env_name, "").strip():
        return None
    return _secret_from_env_path(env_name)


def _grace_open(name: str, now: datetime) -> bool:
    deadline = _grace_deadline(name)
    if deadline is None:
        return False
    return now < deadline


def _grace_deadline(name: str) -> datetime | None:
    raw_path = os.environ.get(f"{name}_GRACE_UNTIL_FILE", "").strip()
    if not raw_path:
        return None
    try:
        text = Path(raw_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    try:
        return _parse_deadline(text)
    except ValueError:
        return None


def _parse_deadline(text: str) -> datetime:
    line = text.strip()
    if len(line) > 64:
        raise ValueError("deadline is too long")
    if line.endswith("Z"):
        line = line[:-1] + "+00:00"
    parsed = datetime.fromisoformat(line)
    if parsed.tzinfo is None:
        raise ValueError("deadline needs a timezone")
    return parsed.astimezone(UTC)


def _secret_from_env_path(env_name: str) -> str:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return _UNREADABLE_SECRET
    try:
        return _read_secret_file(Path(raw))
    except (OSError, UnicodeError, ValueError):
        return _UNREADABLE_SECRET


def _read_secret_file(path: Path) -> str:
    data = path.read_bytes()
    if len(data) > _SECRET_FILE_BYTES:
        raise ValueError("secret file is too large")
    lines = data.decode("utf-8").splitlines()
    if len(lines) != 1 or lines[0] == "":
        raise ValueError("secret file must be a single line")
    return lines[0]


def rate_backend_env_names() -> tuple[str, ...]:
    """Environment names for the shared rate-limit store."""
    return _rate_backend_env_names()


def loopback_rate_env_names() -> tuple[str, ...]:
    """Environment names for the separate loopback buckets."""
    return (
        "QUANT_API_LOOPBACK_RATE_PER_SEC",
        "QUANT_API_LOOPBACK_RATE_BURST",
        "RESEARCH_API_LOOPBACK_RATE_PER_SEC",
        "RESEARCH_API_LOOPBACK_RATE_BURST",
    )


def _rate_pair(
    rate_name: str,
    burst_name: str,
    default_rate: float,
    default_burst: float,
) -> tuple[float, float] | None:
    raw = os.environ.get(rate_name, str(int(default_rate))).strip().lower()
    burst_raw = os.environ.get(burst_name, str(int(default_burst))).strip().lower()
    if raw in _DISABLED or burst_raw in _DISABLED:
        return None
    try:
        rate = float(raw)
        burst = float(burst_raw)
    except ValueError:
        return default_rate, default_burst
    if rate <= 0 or burst <= 0:
        return None
    return rate, burst


def rate_config(prefix: str) -> tuple[float, float] | None:
    """``{PREFIX}_RATE_PER_SEC`` / ``{PREFIX}_RATE_BURST``.

    ``0``, ``off``, ``false``, or ``no`` disables the limit. A value that
    does not parse keeps the default of 10 requests/second and a burst of 30.
    Loopback peers use ``loopback_rate_config`` instead of this pair.
    """
    return _rate_pair(f"{prefix}_RATE_PER_SEC", f"{prefix}_RATE_BURST", 10.0, 30.0)


def loopback_rate_config(prefix: str) -> tuple[float, float] | None:
    """``{PREFIX}_LOOPBACK_RATE_PER_SEC`` / ``{PREFIX}_LOOPBACK_RATE_BURST``.

    The default is 100 requests/second and a burst of 1000. The same disable
    words as ``rate_config`` turn this bucket off without changing the remote one.
    """
    return _rate_pair(
        f"{prefix}_LOOPBACK_RATE_PER_SEC",
        f"{prefix}_LOOPBACK_RATE_BURST",
        100.0,
        1000.0,
    )


def _rate_for_peer(prefix: str, host: str) -> tuple[float, float] | None:
    if is_loopback_host(host):
        return loopback_rate_config(prefix)
    return rate_config(prefix)


def _gate(prefix: str) -> _Gate:
    gate = _GATES.get(prefix)
    if gate is None:
        gate = _Gate()
        _GATES[prefix] = gate
    return gate


def take_rate_token(prefix: str, identity: str, rate: float, burst: float) -> bool | None:
    """Take one token. ``None`` means the selected store cannot be used.

    ``memory`` keeps ``time.monotonic`` so one process can freeze its clock.
    ``sqlite`` and ``file`` use ``time.time`` so two workers share a clock.
    """
    kind = backend_name()
    if kind == "memory":
        return _gate(prefix).allow(identity, rate, burst, time.monotonic())
    store = os.environ.get("DIPCATCHER_RATE_STORE", "").strip()
    if not store:
        return None
    return shared_take(kind, store, identity, rate, burst, time.time())


def _too_many() -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "rate limit exceeded"},
        headers={"Retry-After": "1"},
    )


def _token_response(allowed: bool | None) -> JSONResponse | None:
    if allowed is None:
        return JSONResponse(status_code=503, content={"detail": "rate store unavailable"})
    if not allowed:
        return _too_many()
    return None


def ip_rate_response(request: Request, prefix: str) -> JSONResponse | None:
    host = client_host(request)
    config = _rate_for_peer(prefix, host)
    if config is None:
        return None
    rate, burst = config
    return _token_response(take_rate_token(prefix, f"{prefix}:ip:{host}", rate, burst))


def key_rate_response(request: Request, prefix: str, key: str) -> JSONResponse | None:
    """Per-key bucket. The bucket id is a digest, not the key.

    Loopback and other peers keep separate buckets, so the higher loopback
    limit is not the remote bucket with a new label. A shared store uses
    that same identity.
    """
    host = client_host(request)
    config = _rate_for_peer(prefix, host)
    if config is None:
        return None
    rate, burst = config
    try:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        return None
    scope = "loopback" if is_loopback_host(host) else "remote"
    identity = f"{prefix}:key:{scope}:{digest}"
    return _token_response(take_rate_token(prefix, identity, rate, burst))


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
        request.state.bounded_body = b""
        return
    chunks: list[bytes] = []
    while True:
        message = await original()
        if message.get("type") != "http.request":
            payload = b"".join(chunks)
            request.state.bounded_body = payload
            _replace_receive(request, payload, original, message)
            return
        body = message.get("body", b"") or b""
        if body:
            chunks.append(body)
        if not message.get("more_body", False):
            break
    payload = b"".join(chunks)
    request.state.bounded_body = payload
    _replace_receive(request, payload, original, None)


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
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    for banner in ("server", "x-powered-by"):
        if banner in response.headers:
            del response.headers[banner]
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
