"""Signed job-completion webhooks — sign + verify for the harness's
``callback_url`` deliveries.

When a job submission carries ``callback_secret``, the POSTed job record
arrives with two headers:

- ``X-Fx1-Webhook-Timestamp`` — unix seconds at delivery time
- ``X-Fx1-Webhook-Signature`` — ``sha256=<hmac-sha256 hex>`` over
  ``<timestamp>.<raw body>``

Receivers authenticate a delivery with :func:`verify_webhook` over the
*raw* request body (no re-serialization — signature bytes are exact).
Signing and verification both reject timestamps outside the tolerance
window. The signature is the nonce: a process-local cache refuses to
mint it twice and refuses to accept it twice.
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import ipaddress
import math
import os
import socket
import ssl
import threading
import urllib.parse
from collections import OrderedDict
from typing import NoReturn, cast

from quant_fund.utils.outbound import (
    PRIVATE_ALLOW_CIDRS_ENV,
    OutboundRefused,
    literal_connect_address,
    parse_allow_cidrs,
    vetted_addresses,
)

__all__ = [
    "WEBHOOK_BACKOFF_S",
    "WEBHOOK_MAX_ATTEMPTS",
    "WEBHOOK_SIGNATURE_HEADER",
    "WEBHOOK_TIMESTAMP_HEADER",
    "check_callback_url",
    "deliver_signed",
    "reset_webhook_replay_cache",
    "sign_webhook",
    "verify_webhook",
    "webhook_signature",
]

WEBHOOK_TIMESTAMP_HEADER = "X-Fx1-Webhook-Timestamp"
WEBHOOK_SIGNATURE_HEADER = "X-Fx1-Webhook-Signature"
WEBHOOK_MAX_ATTEMPTS = 3
WEBHOOK_BACKOFF_S = 0.5

_DEFAULT_TOLERANCE_S = 300.0  # 5 min — rejects replayed stale deliveries
_MAX_NONCES = 1024


def _webhook_networks() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    return parse_allow_cidrs(os.environ.get(PRIVATE_ALLOW_CIDRS_ENV, ""))


def _raise_webhook_resolution(host: str, exc: OutboundRefused) -> NoReturn:
    if str(exc) == "allowlist CIDR is invalid":
        raise exc
    if isinstance(exc.__cause__, OSError) or str(exc) in {
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


def _resolved_addresses(host: str, port: int) -> tuple[str, ...]:
    """Classify a literal before DNS and return numeric addresses to dial."""
    try:
        return vetted_addresses(host, port, _webhook_networks(), resolve_fn=socket.getaddrinfo)
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


class _NonceCache:
    """Bounded set of signatures already minted or already accepted."""

    def __init__(self) -> None:
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def remember(self, nonce: str, *, expiry: float, now: float) -> bool:
        """Return true when ``nonce`` was not already present."""
        with self._lock:
            self._prune(now)
            if nonce in self._seen:
                return False
            self._seen[nonce] = expiry
            self._seen.move_to_end(nonce)
            while len(self._seen) > _MAX_NONCES:
                self._seen.popitem(last=False)
            return True

    def _prune(self, now: float) -> None:
        if not math.isfinite(now):
            return
        stale = [key for key, until in self._seen.items() if math.isfinite(until) and until < now]
        for key in stale:
            del self._seen[key]

    def clear(self) -> None:
        with self._lock:
            self._seen.clear()


_ISSUED = _NonceCache()
_ACCEPTED = _NonceCache()


def reset_webhook_replay_cache() -> None:
    """Drop minted and accepted webhook nonces. Tests use this between cases."""
    _ISSUED.clear()
    _ACCEPTED.clear()


def webhook_signature(secret: str, timestamp: str, body: bytes) -> str:
    """Return ``sha256=<hex>`` over ``<timestamp>.<body>`` without touching the nonce cache."""
    digest = hmac.new(
        secret.encode("utf-8"), _signed_payload(timestamp, body), hashlib.sha256
    ).hexdigest()
    return f"sha256={digest}"


def _reference_time(now: float | None) -> float:
    import time  # noqa: PLC0415 — local import keeps the module leaf

    if now is None:
        return time.time()
    return now


def _timestamp_within_tolerance(timestamp: str, tolerance_s: float, now: float | None) -> bool:
    ts = float(timestamp)
    if not math.isfinite(ts) or not math.isfinite(tolerance_s):
        return False
    if tolerance_s < 0:
        return True
    ref = _reference_time(now)
    if not math.isfinite(ref):
        return False
    return abs(ref - ts) <= tolerance_s


def _nonce_expiry(timestamp: str, tolerance_s: float) -> float:
    if tolerance_s < 0:
        return math.inf
    return float(timestamp) + tolerance_s


def _validate_sign_inputs(secret: str, timestamp: str, body: bytes) -> None:
    if not isinstance(secret, str) or not secret:
        raise ValueError("webhook secret is missing")
    if not isinstance(body, bytes):
        raise ValueError("webhook body must be bytes")
    if not isinstance(timestamp, str) or not timestamp or not timestamp.isascii():
        raise ValueError("webhook timestamp is malformed")


def _require_fresh_timestamp(timestamp: str, tolerance_s: float, now: float | None) -> None:
    try:
        fresh = _timestamp_within_tolerance(timestamp, tolerance_s, now)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("webhook timestamp is malformed") from exc
    if not fresh:
        raise ValueError("webhook timestamp is outside the tolerance window")


def sign_webhook(
    secret: str,
    timestamp: str,
    body: bytes,
    *,
    tolerance_s: float = _DEFAULT_TOLERANCE_S,
    now: float | None = None,
) -> str:
    """Mint ``sha256=<hex>`` for one delivery.

    ``timestamp`` must be finite and, unless ``tolerance_s`` is negative,
    inside the tolerance window around ``now`` (default: the current
    clock). The signature is remembered as a nonce; minting it again
    raises ``ValueError`` until it ages out of the window or the bounded
    cache evicts it. The error text never includes the secret.
    """
    _validate_sign_inputs(secret, timestamp, body)
    _require_fresh_timestamp(timestamp, tolerance_s, now)
    signature = webhook_signature(secret, timestamp, body)
    expiry = _nonce_expiry(timestamp, tolerance_s)
    if not _ISSUED.remember(signature, expiry=expiry, now=_reference_time(now)):
        raise ValueError("webhook nonce already issued")
    return signature


def _verify_material(
    secret: str,
    timestamp: str | None,
    signature: str | None,
    body: bytes,
) -> bool:
    if not isinstance(secret, str) or not secret:
        return False
    if not isinstance(timestamp, str) or not timestamp or not timestamp.isascii():
        return False
    if not isinstance(signature, str) or not signature.isascii():
        return False
    if not signature.startswith("sha256="):
        return False
    return isinstance(body, bytes)


def _authenticated(
    secret: str,
    timestamp: str,
    signature: str,
    body: bytes,
    tolerance_s: float,
    now: float | None,
) -> bool:
    if not _timestamp_within_tolerance(timestamp, tolerance_s, now):
        return False
    expected = webhook_signature(secret, timestamp, body)
    if not hmac.compare_digest(expected, signature):
        return False
    expiry = _nonce_expiry(timestamp, tolerance_s)
    return _ACCEPTED.remember(signature, expiry=expiry, now=_reference_time(now))


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
    A signature that has already been accepted is a replay and returns
    ``False``. Any malformed input is a plain ``False`` — never an exception.
    """
    if not _verify_material(secret, timestamp, signature, body):
        return False
    try:
        return _authenticated(
            secret,
            cast(str, timestamp),
            cast(str, signature),
            body,
            tolerance_s,
            now,
        )
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
            f"callback_url must be an http(s) URL with a host and no "
            f"userinfo credentials, got {url!r}"
        )
    try:
        if parsed.port == 0:
            raise ValueError("port zero is not a callback destination")
    except ValueError as exc:
        raise ValueError(f"callback_url has an invalid port, got {url!r}") from exc
    _validate_literal_host(parsed.hostname)
    return url


def deliver_signed(
    url: str,
    secret: str | None,
    body: bytes,
    *,
    max_attempts: int = WEBHOOK_MAX_ATTEMPTS,
    backoff_s: float = WEBHOOK_BACKOFF_S,
    timeout_s: float = 10.0,
) -> tuple[bool, str | None, int]:
    """POST ``body`` to a webhook ``url`` — HMAC-signed when ``secret`` is
    set. Returns ``(delivered, error, attempts)``.

    Transient faults (network errors, 5xx) retry ``max_attempts`` times
    with capped exponential backoff; a 4xx is a definitive rejection and
    is never retried. Delivery faults return as the error string — this
    helper never raises into its caller's worker."""
    import time  # noqa: PLC0415 — local import keeps the module leaf

    error: str | None = None
    try:
        check_callback_url(url)
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname
        if host is None:
            raise ValueError("callback_url has no host")
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
    except Exception as exc:  # noqa: BLE001 — invalid targets are delivery data
        return False, f"{type(exc).__name__}: {exc}", 0

    for attempt in range(max_attempts):
        if attempt:
            time.sleep(backoff_s * (1 << (attempt - 1)))
        try:
            headers = {"Content-Type": "application/json"}
            if secret:
                ts = str(int(time.time()) + attempt)
                headers[WEBHOOK_TIMESTAMP_HEADER] = ts
                headers[WEBHOOK_SIGNATURE_HEADER] = sign_webhook(secret, ts, body)
            addresses = _resolved_addresses(host, port)
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
            error = f"{type(exc).__name__}: {exc}"
    return False, error, max_attempts
