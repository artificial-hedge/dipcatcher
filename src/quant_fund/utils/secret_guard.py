"""Keep credentials out of logs, tracebacks, and world-readable files.

This is the harness implementation. :mod:`fx1.secret_guard` re-exports it
so importing :mod:`fx1` still installs :func:`install_secret_redaction`.
Uncaught exceptions are redacted only while the process still uses the
default ``sys.excepthook`` (test runners replace that hook).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import stat
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

_SECRET_ENV_NAMES = (
    "ALPACA_API_KEY_ID",
    "ALPACA_API_SECRET_KEY",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "DIPCATCHER_SIGSTORE_ID_TOKEN",
    "FX1_API_KEY",
    "FX1_BYOK_API_KEY",
    "FX1_LOCAL_API_KEY",
    "FX1_SIGNING_KEY",
    "MOONSHOT_API_KEY",
    "QUANT_API_KEY",
    "RESEARCH_API_KEY",
)
_MIN_ENV_SECRET = 12
_KEYSTORE_NAMES = frozenset({"keys.jsonl", "credentials.json"})

_BEARER = re.compile(r"(?i)\bBearer\s+\S+")
_ASSIGNED = re.compile(
    r"(?i)\b("
    r"api[_-]?key|callback_secret|webhook_secret|api_secret|secret_key|"
    r"client_secret|access_token|refresh_token|signing_key|password|"
    r"authorization|apca-api-secret-key|apca-api-key-id|x-api-key"
    r")"
    r"([\"']?\s*[=:]\s*[\"']?)"
    r"(?:Bearer\s+)?"
    r"\S+"
)
_PREFIXED = re.compile(r"\b(?:fx1k_|sk-|whsec-)[A-Za-z0-9_-]{8,}")

_Handle = Callable[[logging.Handler, logging.LogRecord], bool]
_ORIGINAL_HANDLE: _Handle | None = None
_INSTALLED = False


def secrets_equal(left: object, right: object) -> bool:
    """Compare two credential strings in constant time and fixed length.

    Both sides are hashed before :func:`hmac.compare_digest`, so a length
    mismatch cannot raise or short-circuit. Non-strings never match.
    """
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    try:
        left_digest = hashlib.sha256(left.encode("utf-8")).digest()
        right_digest = hashlib.sha256(right.encode("utf-8")).digest()
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(left_digest, right_digest)


def redact_text(text: str) -> str:
    """Replace credential material in ``text`` with ``[redacted]``."""
    if not text:
        return text
    cleaned = text
    for secret in _known_secret_values():
        if secret in cleaned:
            cleaned = cleaned.replace(secret, "[redacted]")
    cleaned = _BEARER.sub("Bearer [redacted]", cleaned)
    cleaned = _ASSIGNED.sub(_assigned, cleaned)
    return _PREFIXED.sub("[redacted]", cleaned)


def _assigned(match: re.Match[str]) -> str:
    return match.group(1) + match.group(2) + "[redacted]"


def _known_secret_values() -> list[str]:
    found: list[str] = []
    for name in _SECRET_ENV_NAMES:
        value = os.environ.get(name)
        if isinstance(value, str) and len(value) >= _MIN_ENV_SECRET:
            found.append(value)
    found.sort(key=len, reverse=True)
    return found


class SecretRedactionFilter(logging.Filter):
    """Scrub one log record. Always returns true so the record is still emitted."""

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "_secrets_redacted", False):
            return True
        record.msg = _rendered(record)
        record.args = ()
        if record.exc_info:
            record.exc_text = redact_text(_format_exception(record.exc_info))
            record.exc_info = None
        elif isinstance(record.exc_text, str):
            record.exc_text = redact_text(record.exc_text)
        if isinstance(record.stack_info, str):
            record.stack_info = redact_text(record.stack_info)
        setattr(record, "_secrets_redacted", True)  # noqa: B010
        return True


def _rendered(record: logging.LogRecord) -> str:
    try:
        rendered = record.getMessage()
    except (TypeError, ValueError):
        rendered = str(record.msg)
    return redact_text(rendered)


def _format_exception(exc_info: object) -> str:
    if not isinstance(exc_info, tuple) or len(exc_info) != 3:
        return ""
    _exc_type, exc, _tb = exc_info
    if not isinstance(exc, BaseException):
        return ""
    return "".join(traceback.format_exception(exc))


_FILTER = SecretRedactionFilter()


def redacting_excepthook(
    exc_type: type[BaseException],
    exc: BaseException,
    tb: TracebackType | None,
) -> None:
    """Write a redacted traceback to stderr. Does not include local variables."""
    del exc_type, tb
    text = "".join(traceback.format_exception(exc))
    sys.stderr.write(redact_text(text))


def install_secret_redaction() -> None:
    """Install the redaction filter on the root logger and every handler."""
    global _INSTALLED, _ORIGINAL_HANDLE
    root = logging.getLogger()
    if _FILTER not in root.filters:
        root.addFilter(_FILTER)
    if not _INSTALLED:
        original = logging.Handler.handle
        _ORIGINAL_HANDLE = original

        def _handle(self: logging.Handler, record: logging.LogRecord) -> bool:
            _FILTER.filter(record)
            return original(self, record)

        setattr(logging.Handler, "handle", _handle)  # noqa: B010
        _INSTALLED = True
    if sys.excepthook is sys.__excepthook__:
        sys.excepthook = redacting_excepthook


def refuse_world_readable(path: Path) -> None:
    """Raise when a credential file is readable by users other than the owner.

    The check is the other-read bit. Non-POSIX hosts have no such bit.
    The error names the file, never its contents.
    """
    if os.name != "posix":
        return
    try:
        mode = path.stat().st_mode
    except OSError:
        return
    if stat.S_IMODE(mode) & stat.S_IROTH:
        raise PermissionError(f"{path.name} is world-readable")


def protect_keystore(path: Path) -> None:
    """Force mode 0600 on a keystore or credential file that already exists."""
    if os.name != "posix" or path.name not in _KEYSTORE_NAMES:
        return
    if path.exists():
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def guard_keystore(path: Path) -> None:
    """Refuse a world-readable keystore before its bytes are read."""
    if path.name not in _KEYSTORE_NAMES or not path.exists():
        return
    refuse_world_readable(path)


def is_keystore_path(path: Path) -> bool:
    """True for credential and API-key files written at runtime."""
    return path.name in _KEYSTORE_NAMES
