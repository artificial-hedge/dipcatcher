"""One or two tokens in an existing ``NAME_FILE``.

Line 1 is the token a process loads into ``NAME``. Line 2, when present, is
the previous token. Both are accepted. The file is read on each check so a
running process sees a replacement without a restart. A symlink is ignored.
"""

from __future__ import annotations

import hmac
import os
import stat

_MAX_BYTES = 4096
_FORBIDDEN = "\"'{}#"


def token_ok(value: str) -> bool:
    """True for one token: no whitespace and none of ``\"'{}#``."""
    if not value or any(char.isspace() for char in value):
        return False
    return not any(char in value for char in _FORBIDDEN)


def parse_overlap(text: str) -> tuple[str, ...] | None:
    """Return one or two tokens, or ``None`` when the text is not that shape."""
    if "\r" in text:
        return None
    parts = [part for part in text.split("\n") if part]
    if not parts or len(parts) > 2:
        return None
    if any(not token_ok(part) for part in parts):
        return None
    return tuple(parts)


def read_secret_text(path: str) -> str | None:
    """Return the file text with one trailing newline removed, or ``None``."""
    raw = _read_regular(path)
    if raw is None:
        return None
    return _strip_trailing_break(raw)


def read_secret_tokens(path: str) -> tuple[str, ...] | None:
    """Tokens in ``path``, or ``None`` when the file is missing or not overlap-shaped."""
    text = read_secret_text(path)
    if text is None:
        return None
    return parse_overlap(text)


def accepted_tokens(name: str) -> tuple[str, ...]:
    """Tokens for ``name``. A set ``NAME_FILE`` wins over the plain value."""
    path = os.environ.get(f"{name}_FILE", "").strip()
    if path:
        return read_secret_tokens(path) or ()
    plain = os.environ.get(name, "").strip()
    if token_ok(plain):
        return (plain,)
    return ()


def credential_matches_any(presented: str | None, expected: tuple[str, ...]) -> bool:
    """Constant-time membership. Every token is compared."""
    if presented is None or not expected:
        return False
    presented_bytes = presented.encode("utf-8")
    matched = False
    for token in expected:
        matched = hmac.compare_digest(presented_bytes, token.encode("utf-8")) or matched
    return matched


def auth_decision(name: str, presented: str | None) -> str:
    """``ok``, ``reject``, or ``unset``.

    ``unset`` means neither the file nor the plain value is configured.
    A configured file that cannot be read is ``reject``.
    """
    file_set = bool(os.environ.get(f"{name}_FILE", "").strip())
    tokens = accepted_tokens(name)
    if not file_set and not tokens:
        return "unset"
    if credential_matches_any(presented, tokens):
        return "ok"
    return "reject"


def _strip_trailing_break(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    return text.rstrip("\r\n")


def _read_regular(path: str) -> bytes | None:
    if not path or os.path.islink(path):
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_BYTES:
            return None
        return os.read(descriptor, info.st_size)
    except OSError:
        return None
    finally:
        os.close(descriptor)
