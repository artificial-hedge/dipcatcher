"""Secret redaction shared by logs, transcripts, session files, and diagnostics.

Two layers cooperate:

* **Exact values** — secrets this process loaded (API keys) are registered and
  replaced wherever they appear, including inside larger strings.
* **Shapes** — credential formats in text the process did not author: provider
  key prefixes, bearer headers, URL userinfo, JWTs, PEM private keys, and
  ``api_key=...`` style assignments.

Receipt digests (64-hex SHA-256), token *counts*, and ordinary hex/base64
payloads are deliberately **not** redacted: research claims must stay
quotable from a receipt hash. Redaction is idempotent and never raises.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping, Sequence
from typing import Any

REDACTED = "[REDACTED]"
MIN_SECRET_LEN = 6

_NAME = (
    r"(?:api[_-]?key|apikey|secret|passw(?:or)?d|passwd|credentials?|"
    r"(?:access|refresh|auth|session|id|bearer)[_-]?token|"
    r"private[_-]?key|signing[_-]?key|client[_-]?secret)"
)
SENSITIVE_KEY_RE = re.compile(rf"(?i)^[a-z0-9_.-]*{_NAME}[a-z0-9_.-]*$")

_AUTH_HEADER = re.compile(
    r"(?i)\b(authorization\s*[:=]\s*(?:(?:bearer|basic|token)\s+)?)([\"']?)([A-Za-z0-9._~+/=\-]{8,})"
)
_BEARER_BARE = re.compile(r"(?i)\b(bearer\s+)([A-Za-z0-9._~+/=\-]{16,})")
_KEYED_VALUE = re.compile(
    rf"(?i)([\"']?[a-z0-9_.-]*{_NAME}[a-z0-9_.-]*[\"']?\s*[:=]\s*[\"']?)([^\s\"'\\,;&)}}\]]{{4,}})"
)
_URL_USERINFO = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)([^\s/@:]+):([^\s/@]+)@")
_URL_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:api[_-]?key|apikey|access[_-]?token|token|secret|password|signature|sig)=)"
    r"([^&\s#]+)"
)
_PEM_PRIVATE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
)
_PROVIDER_KEYS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"\bsk-[A-Za-z0-9_\-]{16,}",
        r"\bgh[pousr]_[A-Za-z0-9]{20,}",
        r"\bgithub_pat_[A-Za-z0-9_]{20,}",
        r"\bxox[abprs]-[A-Za-z0-9\-]{10,}",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bAIza[0-9A-Za-z_\-]{30,}",
        r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}",
    )
)

_FLAG_RE = re.compile(
    r"(?i)^--?[a-z0-9-]*(?:api-?key|token|secret|passw(?:or)?d|passwd|credential|authorization)"
    r"[a-z0-9-]*$"
)


def fingerprint(secret: str) -> str:
    """Presence-only fingerprint: the last four characters, never the value."""
    tail = secret[-4:] if len(secret) >= 4 else secret
    return f"...{tail}"


def is_sensitive_key(name: str) -> bool:
    """True when a mapping key or flag name denotes a credential."""
    return bool(SENSITIVE_KEY_RE.match(name))


class SecretRegistry:
    """Thread-safe set of exact secret values to scrub."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: set[str] = set()

    def register(self, value: str) -> None:
        """Track *value*; values shorter than ``MIN_SECRET_LEN`` are ignored."""
        cleaned = value.strip()
        if len(cleaned) >= MIN_SECRET_LEN:
            with self._lock:
                self._values.add(cleaned)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()

    def values(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._values, key=len, reverse=True))

    def redact(self, text: str) -> str:
        for value in self.values():
            if value in text:
                text = text.replace(value, REDACTED)
        return text


DEFAULT_REGISTRY = SecretRegistry()


def register_secret(value: str) -> None:
    """Register an exact secret with the process-wide registry."""
    DEFAULT_REGISTRY.register(value)


def _sub_keyed(match: re.Match[str]) -> str:
    value = match.group(2)
    if value.startswith("[REDACTED"):
        return match.group(0)
    return f"{match.group(1)}{REDACTED}"


def _sub_auth(match: re.Match[str]) -> str:
    if match.group(3).startswith("[REDACTED"):
        return match.group(0)
    return f"{match.group(1)}{match.group(2)}{REDACTED}"


def _redact_shapes(text: str) -> str:
    text = _PEM_PRIVATE.sub(REDACTED, text)
    text = _AUTH_HEADER.sub(_sub_auth, text)
    text = _BEARER_BARE.sub(lambda m: f"{m.group(1)}{REDACTED}", text)
    text = _URL_USERINFO.sub(lambda m: f"{m.group(1)}{REDACTED}@", text)
    text = _URL_QUERY_SECRET.sub(lambda m: f"{m.group(1)}{REDACTED}", text)
    text = _KEYED_VALUE.sub(_sub_keyed, text)
    for pattern in _PROVIDER_KEYS:
        text = pattern.sub(REDACTED, text)
    return text


def redact_text(text: str, *, registry: SecretRegistry | None = None) -> str:
    """Scrub registered secrets and credential shapes from *text*."""
    if not text:
        return text
    scrubbed = (registry or DEFAULT_REGISTRY).redact(text)
    return _redact_shapes(scrubbed)


def contains_secret(text: str, *, registry: SecretRegistry | None = None) -> bool:
    """True when ``redact_text`` would change *text* (used by leak assertions)."""
    return redact_text(text, registry=registry) != text


def _is_blank(value: Any) -> bool:
    return value is None or value is False or (isinstance(value, str) and not value)


def redact_value(obj: Any, *, registry: SecretRegistry | None = None) -> Any:
    """Recursively redact strings; whole values under sensitive keys are masked."""
    if isinstance(obj, str):
        return redact_text(obj, registry=registry)
    if isinstance(obj, Mapping):
        out: dict[Any, Any] = {}
        for key, value in obj.items():
            if isinstance(key, str) and is_sensitive_key(key) and not _is_blank(value):
                out[key] = REDACTED
            else:
                out[key] = redact_value(value, registry=registry)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact_value(item, registry=registry) for item in obj]
    return obj


def redact_argv(argv: Sequence[str], *, registry: SecretRegistry | None = None) -> list[str]:
    """Return *argv* with credential flag values and secret shapes masked."""
    out: list[str] = []
    mask_next = False
    for arg in argv:
        if mask_next:
            out.append(REDACTED)
            mask_next = False
            continue
        flag, sep, value = arg.partition("=")
        if _FLAG_RE.match(flag):
            if sep:
                out.append(f"{flag}={REDACTED}")
            else:
                out.append(arg)
                mask_next = True
            continue
        out.append(redact_text(arg, registry=registry))
    return out
