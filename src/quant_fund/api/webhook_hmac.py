"""HMAC for the alert webhook secret.

The header value is ``sha256=<hex>`` over ``timestamp.body``, the same
construction the harness uses for a signed callback. Overlap verification
tries every configured secret and does not stop at the first match.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence

_TOLERANCE_S = 300.0


def sign_alert_webhook(secret: str, timestamp: str, body: bytes) -> str:
    """Return the signature header value. Rejects an empty secret."""
    if not secret or not timestamp.isascii():
        raise ValueError("webhook material")
    digest = hmac.new(
        secret.encode("utf-8"),
        timestamp.encode("ascii") + b"." + body,
        hashlib.sha256,
    ).hexdigest()
    return "sha256=" + digest


def verify_alert_webhook(
    secrets: Sequence[str],
    timestamp: str | None,
    signature: str | None,
    body: bytes,
    *,
    now: float,
) -> bool:
    """True when one secret signs ``body`` and the timestamp is within five minutes."""
    if not secrets or not isinstance(timestamp, str) or not timestamp.isascii():
        return False
    if not isinstance(signature, str) or not signature.startswith("sha256="):
        return False
    try:
        stamp = float(timestamp)
    except ValueError:
        return False
    if abs(now - stamp) > _TOLERANCE_S:
        return False
    matched = False
    for secret in secrets:
        if not secret:
            continue
        try:
            expected = sign_alert_webhook(secret, timestamp, body)
        except ValueError:
            continue
        matched = hmac.compare_digest(expected, signature) or matched
    return matched
