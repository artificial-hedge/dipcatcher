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
_KEY_AT = re.compile(rf"(?i)(?<![A-Za-z0-9])(?:{_NAME})")
_ASSIGN_TAIL = re.compile(r"""\s*[:=]\s*([\"']?)([^\s\"'\\,;&)}}\]]{4,})""")
_IDENT_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
_JSON_LITERALS = frozenset({"true", "false", "null", "none"})
_UESC = re.compile(r"\\u([0-9a-fA-F]{4})")
# Userinfo runs to the last "@" before the path so a password containing "@" is not split.
_URL_USERINFO = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)([^\s/]+)@")
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
        r"\bsk-(?:[A-Za-z0-9_\-]{16,}|[A-Za-z0-9_\-]{8,}\r?\n[A-Za-z0-9_\-]{8,})",
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


def _sub_auth(match: re.Match[str]) -> str:
    if match.group(3).startswith("[REDACTED"):
        return match.group(0)
    return f"{match.group(1)}{match.group(2)}{REDACTED}"


def _public_literal(value: str) -> bool:
    return value.lower() in _JSON_LITERALS


def _extend_key(text: str, start: int, end: int) -> tuple[int, int]:
    while start > 0 and text[start - 1] in _IDENT_CHARS:
        start -= 1
    if start > 0 and text[start - 1] in "\"'":
        start -= 1
    while end < len(text) and text[end] in _IDENT_CHARS:
        end += 1
    if end < len(text) and text[end] in "\"'":
        end += 1
    return start, end


def _keyed_value_span(text: str, end_name: int) -> tuple[int, int] | None:
    tail = _ASSIGN_TAIL.match(text, end_name)
    if tail is None:
        return None
    value = tail.group(2)
    if value.startswith("[REDACTED") or _public_literal(value):
        return None
    return tail.start(2), tail.end(2)


def _keyed_spans(text: str) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    cursor = 0
    for match in _KEY_AT.finditer(text):
        if match.start() < cursor:
            continue
        _start, end_name = _extend_key(text, match.start(), match.end())
        span = _keyed_value_span(text, end_name)
        if span is None:
            continue
        found.append(span)
        cursor = span[1]
    return found


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    ordered = sorted(spans)
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _replace_spans(text: str, spans: list[tuple[int, int]]) -> str:
    pieces: list[str] = []
    cursor = 0
    for start, end in _merge_spans(spans):
        if end <= cursor:
            continue
        begin = max(start, cursor)
        pieces.append(text[cursor:begin])
        pieces.append(REDACTED)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _redact_keyed(text: str) -> str:
    return _replace_spans(text, _keyed_spans(text))


def _literal_spans(text: str, values: tuple[str, ...]) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    for value in values:
        start = 0
        while True:
            idx = text.find(value, start)
            if idx < 0:
                break
            found.append((idx, idx + len(value)))
            start = idx + len(value)
    return found


def _pattern_spans(text: str) -> list[tuple[int, int]]:
    found = [match.span() for match in _PEM_PRIVATE.finditer(text)]
    found.extend(match.span() for match in _BEARER_BARE.finditer(text))
    found.extend(match.span() for match in _URL_USERINFO.finditer(text))
    found.extend(match.span() for match in _URL_QUERY_SECRET.finditer(text))
    found.extend(match.span() for match in _AUTH_HEADER.finditer(text))
    for pattern in _PROVIDER_KEYS:
        found.extend(match.span() for match in pattern.finditer(text))
    return found


def _decode_ascii_u(text: str) -> tuple[str, list[tuple[int, int]]] | None:
    """Map JSON ``\\u00XX`` ASCII escapes to characters, with original spans."""
    if "\\u" not in text:
        return None
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    index = 0
    changed = False
    while index < len(text):
        match = _UESC.match(text, index)
        if match is not None and 32 <= int(match.group(1), 16) < 127:
            parts.append(chr(int(match.group(1), 16)))
            spans.append((match.start(), match.end()))
            index = match.end()
            changed = True
            continue
        parts.append(text[index])
        spans.append((index, index + 1))
        index += 1
    if not changed:
        return None
    return "".join(parts), spans


def _orig_span(spans: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    return spans[start][0], spans[end - 1][1]


def _redact_unicode_secrets(text: str, registry: SecretRegistry) -> str:
    decoded = _decode_ascii_u(text)
    if decoded is None:
        return text
    view, spans = decoded
    hits = _keyed_spans(view)
    hits.extend(_literal_spans(view, registry.values()))
    hits.extend(_pattern_spans(view))
    if not hits:
        return text
    orig = [_orig_span(spans, start, end) for start, end in hits if end > start]
    return _replace_spans(text, orig)


def _redact_shapes(text: str) -> str:
    text = _PEM_PRIVATE.sub(REDACTED, text)
    text = _AUTH_HEADER.sub(_sub_auth, text)
    text = _BEARER_BARE.sub(lambda m: f"{m.group(1)}{REDACTED}", text)
    text = _URL_USERINFO.sub(lambda m: f"{m.group(1)}{REDACTED}@", text)
    text = _URL_QUERY_SECRET.sub(lambda m: f"{m.group(1)}{REDACTED}", text)
    text = _redact_keyed(text)
    for pattern in _PROVIDER_KEYS:
        text = pattern.sub(REDACTED, text)
    return text


def redact_text(text: str, *, registry: SecretRegistry | None = None) -> str:
    """Scrub registered secrets and credential shapes from *text*."""
    if not text:
        return text
    reg = registry or DEFAULT_REGISTRY
    scrubbed = reg.redact(text)
    scrubbed = _redact_unicode_secrets(scrubbed, reg)
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
