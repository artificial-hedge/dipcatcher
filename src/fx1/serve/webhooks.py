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
import math

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
    import urllib.parse  # noqa: PLC0415 — local import keeps the module leaf

    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            f"callback_url must be an http(s) URL with a host and no "
            f"userinfo credentials, got {url!r}"
        )
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
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    error: str | None = None
    for attempt in range(max_attempts):
        if attempt:
            time.sleep(backoff_s * (1 << (attempt - 1)))
        try:
            headers = {"Content-Type": "application/json"}
            if secret:
                ts = str(int(time.time()))
                headers[WEBHOOK_TIMESTAMP_HEADER] = ts
                headers[WEBHOOK_SIGNATURE_HEADER] = sign_webhook(secret, ts, body)
            req = urllib.request.Request(
                url,
                data=body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310  # nosec B310 — caller-declared webhook target, validated http(s) at submit
                if resp.status < 400:
                    return True, None, attempt + 1
                error = f"callback endpoint returned {resp.status}"
                if 400 <= resp.status < 500:
                    return False, error, attempt + 1
        except urllib.error.HTTPError as exc:
            # urlopen raises on any >=400 — the status is still the verdict:
            # 4xx is a definitive rejection (never retried), 5xx is transient.
            error = f"callback endpoint returned {exc.code}"
            if 400 <= exc.code < 500:
                return False, error, attempt + 1
        except Exception as exc:  # noqa: BLE001 — delivery faults are data, never raised
            error = f"{type(exc).__name__}: {exc}"
    return False, error, max_attempts
