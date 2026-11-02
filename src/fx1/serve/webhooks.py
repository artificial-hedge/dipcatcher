"""Signed job-completion webhooks — sign + verify for the harness's
``callback_url`` deliveries.

When a job submission carries ``callback_secret``, the POSTed job record
arrives with two headers:

- ``X-Fx1-Webhook-Timestamp`` — unix seconds at delivery time
- ``X-Fx1-Webhook-Signature`` — ``sha256=<hmac-sha256 hex>`` over
  ``<timestamp>.<raw body>``

Receivers authenticate a delivery with :func:`verify_webhook` over the
*raw* request body (no re-serialization — signature bytes are exact).
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import ipaddress
import math
import os
import random
import socket
import ssl
import urllib.parse
from typing import Any, NoReturn

from quant_fund.utils.net_timeout import bounded_getaddrinfo, require_attempts, require_timeout
from quant_fund.utils.outbound import (
    PRIVATE_ALLOW_CIDRS_ENV,
    OutboundRefused,
    literal_connect_address,
    parse_allow_cidrs,
    vetted_addresses,
)
from quant_fund.utils.secrets import redact_text

__all__ = [
    "WEBHOOK_BACKOFF_S",
    "WEBHOOK_MAX_ATTEMPTS",
    "WEBHOOK_SIGNATURE_HEADER",
    "WEBHOOK_TIMESTAMP_HEADER",
    "check_callback_url",
    "deliver_signed",
    "sign_webhook",
    "verify_webhook",
]

WEBHOOK_TIMESTAMP_HEADER = "X-Fx1-Webhook-Timestamp"
WEBHOOK_SIGNATURE_HEADER = "X-Fx1-Webhook-Signature"
WEBHOOK_MAX_ATTEMPTS = 3
WEBHOOK_BACKOFF_S = 0.5

_DEFAULT_TOLERANCE_S = 300.0  # 5 min — rejects replayed stale deliveries


def _webhook_networks() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    return parse_allow_cidrs(os.environ.get(PRIVATE_ALLOW_CIDRS_ENV, ""))


def _raise_webhook_resolution(host: str, exc: OutboundRefused) -> NoReturn:
    if str(exc) == "allowlist CIDR is invalid":
        raise exc
    cause = exc.__cause__
    if isinstance(cause, TimeoutError):
        raise TimeoutError(str(cause)) from exc
    if isinstance(cause, OSError) or str(exc) in {
        "cannot resolve host",
        "destination host did not resolve to an IP address",
    }:
        raise OSError(f"callback_url host {host!r} did not resolve to an IP address") from exc
    raise ValueError(f"callback_url resolved to a private or special-use address: {exc}") from exc


def _validate_literal_host(host: str) -> None:
    """Reject a special-use IP literal without performing DNS at submit time."""
    try:
        literal_connect_address(host, _webhook_networks())
    except OutboundRefused as exc:
        raise ValueError("callback_url must not target a private or special-use address") from exc


def _resolved_addresses(host: str, port: int, timeout: float = 10.0) -> tuple[str, ...]:
    """Classify a literal before DNS and return numeric addresses to dial."""

    def resolve(name: str, number: int) -> list[tuple[Any, ...]]:
        return bounded_getaddrinfo(
            name,
            number,
            timeout,
            type=socket.SOCK_STREAM,
            lookup=socket.getaddrinfo,
        )

    try:
        return vetted_addresses(host, port, _webhook_networks(), resolve_fn=resolve)
    except OutboundRefused as exc:
        _raise_webhook_resolution(host, exc)


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection whose TCP destination is the validated numeric address."""

    source_address: tuple[str, int] | None

    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        super().__init__(host, port=port, timeout=timeout)
        self._address = address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._address, self.port),
            self.timeout,
            self.source_address,
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection pinned to an IP while authenticating the URL host."""

    source_address: tuple[str, int] | None
    _context: ssl.SSLContext

    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        context = ssl.create_default_context()
        # Explicit contexts must advertise the protocol used by http.client.
        context.set_alpn_protocols(["http/1.1"])
        super().__init__(
            host,
            port=port,
            timeout=timeout,
            context=context,
        )
        self._address = address

    def connect(self) -> None:
        raw_sock = socket.create_connection(
            (self._address, self.port),
            self.timeout,
            self.source_address,
        )
        self.sock = self._context.wrap_socket(raw_sock, server_hostname=self.host)


def _post_once(
    parsed: urllib.parse.ParseResult,
    address: str,
    body: bytes,
    headers: dict[str, str],
    timeout_s: float,
) -> int:
    """POST once to a previously validated address; never follow redirects."""
    host = parsed.hostname
    if host is None:  # defensive; check_callback_url rejects this first
        raise ValueError("callback_url has no host")
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    connection_cls = _PinnedHTTPSConnection if parsed.scheme == "https" else _PinnedHTTPConnection
    connection = connection_cls(host, port, address, timeout_s)
    target = urllib.parse.urlunparse(("", "", parsed.path or "/", parsed.params, parsed.query, ""))
    try:
        connection.request("POST", target, body=body, headers=headers)
        # Delivery depends only on status; an untrusted response body may be
        # unbounded. This connection is never reused, so close without draining.
        with connection.getresponse() as response:
            return response.status
    finally:
        connection.close()


def _signed_payload(timestamp: str, body: bytes) -> bytes:
    return timestamp.encode("ascii") + b"." + body


def sign_webhook(secret: str, timestamp: str, body: bytes) -> str:
    """Return ``sha256=<hex>`` — the signature header value."""
    sig = hmac.new(
        secret.encode("utf-8"), _signed_payload(timestamp, body), hashlib.sha256
    ).hexdigest()
    return f"sha256={sig}"


def verify_webhook(
    secret: str,
    timestamp: str | None,
    signature: str | None,
    body: bytes,
    *,
    tolerance_s: float = _DEFAULT_TOLERANCE_S,
    now: float | None = None,
) -> bool:
    """Authenticate a webhook delivery using a constant-time signature compare.

    Timestamps and tolerances must be finite. Nonnegative ``tolerance_s``
    allows at most that many seconds of past or future clock skew (inclusive);
    a finite negative tolerance explicitly disables the freshness check.
    Any malformed input is a plain ``False`` — never an exception.
    """
    if not isinstance(secret, str) or not secret:
        return False
    if not isinstance(timestamp, str) or not timestamp or not timestamp.isascii():
        return False
    if not isinstance(signature, str) or not signature.isascii():
        return False
    if not signature.startswith("sha256="):
        return False
    try:
        ts = float(timestamp)
        if not math.isfinite(ts) or not math.isfinite(tolerance_s):
            return False
        if tolerance_s >= 0:
            import time  # noqa: PLC0415 — local import keeps the module leaf

            ref = time.time() if now is None else now
            if not math.isfinite(ref) or abs(ref - ts) > tolerance_s:
                return False
        expected = sign_webhook(secret, timestamp, body)
        return hmac.compare_digest(expected, signature)
    except (TypeError, ValueError, OverflowError):
        # Includes encoding failures (UnicodeError is a ValueError subclass).
        return False


def check_callback_url(url: str | None) -> str | None:
    """The shared ``callback_url`` field validator — a webhook target must
    be a real http(s) URL with a host. Returns ``url`` unchanged; raises
    ``ValueError`` on anything else."""
    if url is None:
        return url
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "callback_url must be an http(s) URL with a host and no userinfo credentials"
        )
    try:
        if parsed.port == 0:
            raise ValueError("port zero is not a callback destination")
    except ValueError as exc:
        raise ValueError("callback_url has an invalid port") from exc
    _validate_literal_host(parsed.hostname)
    return url


def _delivery_arg_error(max_attempts: int, backoff_s: float, timeout_s: float) -> str | None:
    if max_attempts < 1:
        return "ValueError: max_attempts must be positive"
    if not math.isfinite(backoff_s) or backoff_s < 0:
        return "ValueError: backoff_s must be finite and non-negative"
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        return "ValueError: timeout_s must be finite and positive"
    return None


def _callback_endpoint(url: str) -> tuple[urllib.parse.ParseResult, str, int]:
    check_callback_url(url)
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    if host is None:
        raise ValueError("callback_url has no host")
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return parsed, host, port


def _retry_delay(
    attempt: int,
    backoff_s: float,
    *,
    jitter: bool,
    rng: random.Random | None,
) -> float:
    """Capped exponential backoff. Jitter is drawn from ``[0, cap]``."""
    delay = min(30.0, backoff_s * (1 << (attempt - 1)))
    if jitter and delay > 0:
        uniform = rng.uniform if rng is not None else random.uniform
        return uniform(0.0, delay)
    return delay


def deliver_signed(
    url: str,
    secret: str | None,
    body: bytes,
    *,
    max_attempts: int = WEBHOOK_MAX_ATTEMPTS,
    backoff_s: float = WEBHOOK_BACKOFF_S,
    timeout_s: float = 10.0,
    jitter: bool = True,
    rng: random.Random | None = None,
) -> tuple[bool, str | None, int]:
    """POST ``body`` to a webhook ``url`` — HMAC-signed when ``secret`` is
    set. Returns ``(delivered, error, attempts)``.

    Transient faults (network errors, 5xx) retry ``max_attempts`` times
    with bounded exponential backoff and jitter; a 4xx is a definitive
    rejection and is never retried. Delivery faults return as the error
    string — this helper never raises into its caller's worker."""
    import time  # noqa: PLC0415 — local import keeps the module leaf

    arg_error = _delivery_arg_error(max_attempts, backoff_s, timeout_s)
    if arg_error is not None:
        return False, arg_error, 0

    error: str | None = None
    try:
        timeout_s = require_timeout(timeout_s, name="timeout_s")
        max_attempts = require_attempts(max_attempts)
        parsed, host, port = _callback_endpoint(url)
    except Exception as exc:  # noqa: BLE001 — invalid targets are delivery data
        return False, redact_text(f"{type(exc).__name__}: {exc}"), 0

    for attempt in range(max_attempts):
        if attempt:
            time.sleep(_retry_delay(attempt, backoff_s, jitter=jitter, rng=rng))
        try:
            headers = {"Content-Type": "application/json"}
            if secret:
                ts = str(int(time.time()))
                headers[WEBHOOK_TIMESTAMP_HEADER] = ts
                headers[WEBHOOK_SIGNATURE_HEADER] = sign_webhook(secret, ts, body)
            addresses = _resolved_addresses(host, port, timeout_s)
            last_exception: Exception | None = None
            for address in addresses:
                try:
                    status = _post_once(parsed, address, body, headers, timeout_s)
                except Exception as exc:  # noqa: BLE001 — try another validated address
                    last_exception = exc
                    continue
                if status < 300:
                    return True, None, attempt + 1
                error = f"callback endpoint returned {status}"
                if 400 <= status < 500:
                    return False, error, attempt + 1
                break
            else:
                if last_exception is not None:
                    raise last_exception
        except Exception as exc:  # noqa: BLE001 — delivery faults are data, never raised
            error = redact_text(f"{type(exc).__name__}: {exc}")
    return False, error, max_attempts
