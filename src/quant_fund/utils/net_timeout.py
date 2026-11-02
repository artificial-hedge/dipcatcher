"""Bounds for network calls the stdlib leaves open.

``socket.getaddrinfo`` takes no timeout. ``HTTPResponse.read()`` restarts
the socket timeout on every chunk, so a peer that keeps sending never
trips it. ``NaN`` and ``inf`` also pass a ``timeout <= 0`` check. These
helpers close those holes without a new dependency.
"""

from __future__ import annotations

import math
import socket
import threading
import time
from typing import Any

DEFAULT_MAX_BODY = 2_000_000
MAX_TRANSPORT_RETRIES = 8


def require_timeout(value: object, *, name: str = "timeout") -> float:
    """Return ``value`` as a positive finite number of seconds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return timeout


def require_retries(
    value: object, *, name: str = "retries", limit: int = MAX_TRANSPORT_RETRIES
) -> int:
    """Return a retry count in ``0..limit``. Bool is rejected."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > limit:
        raise ValueError(f"{name} must be an int in 0..{limit}, got {value!r}")
    return value


def require_attempts(
    value: object, *, name: str = "max_attempts", limit: int = MAX_TRANSPORT_RETRIES
) -> int:
    """Return an attempt count in ``1..limit``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > limit:
        raise ValueError(f"{name} must be an int in 1..{limit}, got {value!r}")
    return value


def bounded_getaddrinfo(
    host: str,
    port: int | str | None,
    timeout: float,
    *,
    type: int = 0,
) -> list[tuple[Any, ...]]:
    """Resolve ``host`` or raise ``TimeoutError`` when DNS exceeds ``timeout``.

    The lookup runs on a daemon thread. A timed-out resolver call cannot be
    cancelled inside the C library; the thread is abandoned.
    """
    timeout_s = require_timeout(timeout, name="timeout")
    box: list[list[tuple[Any, ...]]] = []
    err: list[Exception] = []
    done = threading.Event()

    def run() -> None:
        try:
            box.append(socket.getaddrinfo(host, port, type=type))
        except Exception as exc:  # noqa: BLE001 — re-raised on the caller thread
            err.append(exc)
        finally:
            done.set()

    threading.Thread(target=run, name="dns-timeout", daemon=True).start()
    if not done.wait(timeout_s):
        raise TimeoutError(f"DNS lookup for {host!r} exceeded {timeout_s:g}s")
    if err:
        raise err[0]
    return box[0]


def read_bounded(response: Any, *, timeout: float, max_bytes: int = DEFAULT_MAX_BODY) -> bytes:
    """Read a response body until EOF, ``max_bytes``, or the deadline.

    ``timeout`` covers the whole read. A no-argument ``read()`` (test
    doubles) is taken once. Exceeding the cap raises ``ValueError``.
    """
    timeout_s = require_timeout(timeout, name="timeout")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError(f"max_bytes must be a positive int, got {max_bytes!r}")
    deadline = time.monotonic() + timeout_s
    chunks: list[bytes] = []
    total = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("response body exceeded the read deadline")
        _tighten_socket_timeout(response, remaining)
        block, chunked = _read_chunk(response)
        if not isinstance(block, (bytes, bytearray)):
            raise ValueError("response body is not bytes")
        if not block:
            return b"".join(chunks)
        total += len(block)
        if total > max_bytes:
            raise ValueError(f"response exceeded {max_bytes} bytes")
        if not chunked:
            return bytes(block)
        chunks.append(bytes(block))


def _read_chunk(response: Any) -> tuple[bytes | bytearray, bool]:
    read = response.read
    try:
        return read(65536), True
    except TypeError:
        return read(), False


def _tighten_socket_timeout(response: Any, remaining: float) -> None:
    """Shorten the underlying socket timeout to the time still left."""
    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is not None and hasattr(sock, "settimeout"):
        sock.settimeout(remaining)


def read_pinned(
    request: Any,
    *,
    timeout: float,
    max_bytes: int = DEFAULT_MAX_BODY,
    allow_loopback: bool = False,
    allow_private: bool = False,
    resolve_fn: Any = None,
) -> bytes:
    """DNS-pinned body read that stops at ``max_bytes`` and the deadline.

    Redirects are refused. Status 400 and above raise ``urllib.error.HTTPError``.
    """
    import urllib.error

    from quant_fund.utils.outbound import OutboundRefused, open_pinned

    timeout_s = require_timeout(timeout, name="timeout")
    with open_pinned(
        request,
        timeout=timeout_s,
        allow_loopback=allow_loopback,
        allow_private=allow_private,
        resolve_fn=resolve_fn,
    ) as handle:
        response = getattr(handle, "_response", handle)
        body = read_bounded(response, timeout=timeout_s, max_bytes=max_bytes)
        status = int(handle.status)
        if 300 <= status < 400:
            raise OutboundRefused("redirects are not followed")
        if status >= 400:
            raise urllib.error.HTTPError(
                handle.geturl(),
                status,
                handle.reason or "error",
                handle.headers,
                None,
            )
        return body
