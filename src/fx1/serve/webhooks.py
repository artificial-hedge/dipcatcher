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

__all__ = [
    "WEBHOOK_SIGNATURE_HEADER",
    "WEBHOOK_TIMESTAMP_HEADER",
    "sign_webhook",
    "verify_webhook",
]

WEBHOOK_TIMESTAMP_HEADER = "X-Fx1-Webhook-Timestamp"
WEBHOOK_SIGNATURE_HEADER = "X-Fx1-Webhook-Signature"

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
    """Authenticate a webhook delivery. Constant-time compare; the
    timestamp must parse and sit within ``tolerance_s`` of ``now`` so a
    captured request can't be replayed later. Any malformed input is a
    plain ``False`` — never an exception."""
    if not secret or not timestamp or not signature:
        return False
    if not signature.startswith("sha256="):
        return False
    try:
        ts = float(timestamp)
    except ValueError:
        return False
    if tolerance_s >= 0:
        import time  # noqa: PLC0415 — local import keeps the module leaf

        ref = time.time() if now is None else now
        if abs(ref - ts) > tolerance_s:
            return False
    expected = sign_webhook(secret, timestamp, body)
    return hmac.compare_digest(expected, signature)
