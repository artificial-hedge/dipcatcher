"""Load API keys and the alert webhook secret from files.

``NAME_FILE`` points at a regular file, including a docker secret mounted
under ``/run/secrets``. Under ``DIPCATCHER_PROFILE`` ``production`` or
``prod``, a plain ``NAME`` value is refused. When ``NAME_FILE`` is set, a
plain ``NAME`` that this process did not load from that file is refused.
Messages name the variable and do not include the value.
"""

from __future__ import annotations

import os
import stat
from urllib.parse import urlsplit

from quant_fund.utils.secret_tokens import parse_overlap

SECRET_NAMES = (
    "QUANT_API_KEY",
    "RESEARCH_API_KEY",
    "DIPCATCHER_ALERT_WEBHOOK_SECRET",
)
_PRODUCTION = frozenset({"production", "prod"})
_MAX_BYTES = 4096
_LOADED: dict[str, str] = {}
_OWNED: set[str] = set()


class SecretFileError(ValueError):
    """A file secret could not be loaded. The message has no secret value."""


def clear_loaded_secrets() -> None:
    """Remove values this process copied out of secret files."""
    for name in list(_OWNED):
        os.environ.pop(name, None)
    _OWNED.clear()
    _LOADED.clear()


def apply_secret_files() -> list[str]:
    """Copy file secrets into the process environment. Return problems."""
    production = _production()
    problems = [problem for name in SECRET_NAMES if (problem := _apply_one(name, production))]
    webhook = _webhook_problem(production)
    if webhook:
        problems.append(webhook)
    return problems


def _production() -> bool:
    return os.environ.get("DIPCATCHER_PROFILE", "").strip().lower() in _PRODUCTION


def _apply_one(name: str, production: bool) -> str | None:
    file_var = f"{name}_FILE"
    path = os.environ.get(file_var, "").strip()
    plain = os.environ.get(name, "").strip()
    if path and plain:
        if _LOADED.get(name) == path:
            return None
        return f"{name} must not be set in the environment when {file_var} is set"
    if path:
        return _load(name, file_var, path)
    if production and plain:
        return f"{name} must be loaded from {file_var} when DIPCATCHER_PROFILE is production"
    return None


def _load(name: str, file_var: str, path: str) -> str | None:
    readable, value = _read_token(path)
    if not readable:
        return f"{file_var} is not a readable file"
    tokens = parse_overlap(value)
    if tokens is None:
        return f"{file_var} must contain one or two tokens"
    os.environ[name] = tokens[0]
    _LOADED[name] = path
    _OWNED.add(name)
    return None


def _read_token(path: str) -> tuple[bool, str]:
    """Return whether ``path`` could be read, and the text with one trailing newline removed."""
    raw = _read_regular(path)
    if raw is None:
        return False, ""
    return True, _strip_secret(raw)


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


def _strip_secret(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    return text.rstrip("\r\n")


def _webhook_problem(production: bool) -> str | None:
    url = os.environ.get("DIPCATCHER_ALERT_WEBHOOK_URL", "").strip()
    if not url:
        return None
    if not _webhook_url_ok(url):
        return "DIPCATCHER_ALERT_WEBHOOK_URL is invalid"
    file_set = bool(os.environ.get("DIPCATCHER_ALERT_WEBHOOK_SECRET_FILE", "").strip())
    plain = bool(os.environ.get("DIPCATCHER_ALERT_WEBHOOK_SECRET", "").strip())
    if file_set or plain:
        return None
    if production:
        return (
            "DIPCATCHER_ALERT_WEBHOOK_SECRET_FILE is required "
            "when DIPCATCHER_ALERT_WEBHOOK_URL is set"
        )
    return "DIPCATCHER_ALERT_WEBHOOK_SECRET is required when DIPCATCHER_ALERT_WEBHOOK_URL is set"


def _webhook_url_ok(url: str) -> bool:
    if len(url) > 512 or any(char.isspace() for char in url):
        return False
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return False
    return not parts.username and not parts.password
