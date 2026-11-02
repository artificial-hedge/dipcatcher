"""Process-wide token bucket for the lab and research APIs.

Unset or ``0`` disables the limit. A positive ``DIPCATCHER_RATE_LIMIT_RPS``
answers overload with HTTP 429. ``/health`` stays unlimited so a probe is not
counted against the caller. Invalid values are refused at startup; if one
still reaches a request, the answer is 429 rather than 500.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Any

_ENV = "DIPCATCHER_RATE_LIMIT_RPS"
_PUBLIC = frozenset({"/health"})
_MAX_RPS = 1_000_000.0
_BODY = b'{"detail":"rate limit exceeded"}\n'


def configured_rps() -> float:
    """Return the limit. ``0`` is off. ``-1`` means the value is unusable."""
    raw = os.environ.get(_ENV, "").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return -1.0
    if not math.isfinite(value) or value < 0.0 or value > _MAX_RPS:
        return -1.0
    return value


class TokenBucket:
    """A one-second burst equal to the configured rate."""

    def __init__(self) -> None:
        self.rps = 0.0
        self.tokens = 0.0
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def take(self) -> tuple[bool, int]:
        """Return whether the request may proceed, and a Retry-After hint."""
        rps = configured_rps()
        if rps == 0.0:
            return True, 0
        if rps < 0.0:
            return False, 1
        now = time.monotonic()
        with self._lock:
            self._refill(rps, now)
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True, 0
            deficit = 1.0 - self.tokens
            return False, max(1, math.ceil(deficit / rps))

    def _refill(self, rps: float, now: float) -> None:
        if rps != self.rps:
            self.rps = rps
            self.tokens = rps
            self.updated = now
            return
        elapsed = now - self.updated
        self.updated = now
        self.tokens = min(rps, self.tokens + elapsed * rps)


BUCKET = TokenBucket()


def reset_bucket() -> None:
    """Drop leftover tokens so a later limit starts from a full burst."""
    with BUCKET._lock:
        BUCKET.rps = 0.0
        BUCKET.tokens = 0.0
        BUCKET.updated = time.monotonic()


def _headers(retry_after: int, rps: float) -> list[tuple[bytes, bytes]]:
    limit = "invalid" if rps < 0 else str(int(rps) if rps.is_integer() else rps)
    return [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(_BODY)).encode()),
        (b"retry-after", str(retry_after).encode()),
        (b"x-ratelimit-limit", limit.encode()),
    ]


class RateLimitMiddleware:
    """ASGI middleware. Health probes are not limited."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") in _PUBLIC:
            await self.app(scope, receive, send)
            return
        allowed, retry_after = BUCKET.take()
        if allowed:
            await self.app(scope, receive, send)
            return
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": _headers(retry_after, configured_rps()),
            }
        )
        await send({"type": "http.response.body", "body": _BODY})
