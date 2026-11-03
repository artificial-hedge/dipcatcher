"""Redact credentials before they reach logs or exception text.

Field names such as ``api_key`` are dropped entirely. Values are scrubbed
when they carry URL userinfo (including DSNs), secret query parameters,
bearer tokens, or known key prefixes. A public URL with no credential
stays readable.
"""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[redacted]"

_SECRET_NAMES = frozenset(
    {
        "key",
        "token",
        "secret",
        "authorization",
        "apikey",
        "api_key",
        "private_key",
        "privatekey",
        "session_key",
        "access_token",
        "dsn",
        "sentry_dsn",
        "callback_url",
        "webhook_url",
        "database_url",
        "redis_url",
        "amqp_url",
    }
)

# ``user:secret@host`` and ``publickey@host`` (Sentry-style DSNs).
_USERINFO = re.compile(r"([a-z][a-z0-9+.-]*://)([^/\s?#]+@)", re.IGNORECASE)
_QUERY_SECRET = re.compile(
    r"([?&](?:api[_-]?key|access[_-]?token|token|secret|password|passwd|"
    r"signature|sig|userid|user_id|dsn|key)=)[^&#\s]+",
    re.IGNORECASE,
)
_BEARER = re.compile(r"(\bBearer\s+)[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE)
_KEY_PREFIX = re.compile(r"\b(?:sk-|fx1k[_-]|whsec-)[A-Za-z0-9_-]{8,}\b")


def is_secret_field(key: str) -> bool:
    """True when a log field name is itself a credential slot."""
    lowered = key.lower().replace("-", "_")
    if "password" in lowered or lowered in _SECRET_NAMES:
        return True
    return lowered.endswith(("_key", "_token", "_secret", "_dsn"))


def redact_text(text: str) -> str:
    """Remove credential material from a string. Non-secrets stay put."""
    scrubbed = _USERINFO.sub(rf"\1{REDACTED}@", text)
    scrubbed = _QUERY_SECRET.sub(rf"\1{REDACTED}", scrubbed)
    scrubbed = _BEARER.sub(rf"\1{REDACTED}", scrubbed)
    return _KEY_PREFIX.sub(REDACTED, scrubbed)


def endpoint_mismatch(where: str, url: object, endpoint: str) -> str:
    """Batch URL error with credentials removed from both sides."""
    return (
        f"{where}: url {redact_text(str(url))!r} does not match the batch endpoint "
        f"{redact_text(endpoint)!r}"
    )


def redact_log_value(key: str, value: Any) -> Any:
    """Redact one log field. Secret names are replaced wholesale."""
    if is_secret_field(key):
        return REDACTED
    return redact_obj(value)


def redact_obj(value: Any) -> Any:
    """Walk strings, mappings, and sequences. Other objects are unchanged."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, BaseException):
        return redact_text(f"{type(value).__name__}: {value}")
    if isinstance(value, dict):
        return {str(key): redact_log_value(str(key), item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_obj(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_obj(item) for item in value)
    return value
