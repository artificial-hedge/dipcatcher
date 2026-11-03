"""Fail fast before the API process is executed.

A symlink is always refused. The write probe runs when
``DIPCATCHER_REQUIRE_DATA_WRITE`` is on, and also when
``DIPCATCHER_PROFILE`` is ``production`` or ``prod``, unless the command is
the research API. Production research must see a data directory it cannot
write. That profile leaves an unset ``DIPCATCHER_RATE_LIMIT_RPS`` at the
process default of 50 so the limit cannot stay off. Zero and any invalid
rate still refuse to boot. The research image leaves the write flag off;
``deploy/prod.env`` turns the profile on. Under that profile, API keys and
the alert webhook secret are read from ``*_FILE`` and a plain value is
refused. This file uses the stdlib only.
"""

from __future__ import annotations

import errno
import math
import os
import stat
import sys
from urllib.parse import urlsplit

_DIP_UID = 10001
_TRUE = frozenset({"1", "true", "yes", "on"})
_PRODUCTION = frozenset({"production", "prod"})
_RATE_MESSAGE = "DIPCATCHER_RATE_LIMIT_RPS must be set above 0 in production"
_PRODUCTION_DEFAULT_RPS = "50"
_RESEARCH_ARGV = "quant_fund.api.research_api"
_READ_ONLY_ERRNO = frozenset({errno.EROFS, errno.EACCES, errno.EPERM})
_RESEARCH_RO = "must be read-only when DIPCATCHER_PROFILE is production"
_SECRET_NAMES = (
    "QUANT_API_KEY",
    "RESEARCH_API_KEY",
    "DIPCATCHER_ALERT_WEBHOOK_SECRET",
)
_SECRET_MAX_BYTES = 4096


def _enabled(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUE


def _production() -> bool:
    return os.environ.get("DIPCATCHER_PROFILE", "").strip().lower() in _PRODUCTION


def _apply_production_rate_default() -> None:
    """Fill an unset production rate so the child process cannot run unlimited."""
    if _production() and not os.environ.get("DIPCATCHER_RATE_LIMIT_RPS", "").strip():
        os.environ["DIPCATCHER_RATE_LIMIT_RPS"] = _PRODUCTION_DEFAULT_RPS


def _production_rate_problem() -> str | None:
    """Return the production rate message, or ``None`` when the boot may continue."""
    if not _production():
        return None
    raw = os.environ.get("DIPCATCHER_RATE_LIMIT_RPS", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return _RATE_MESSAGE
    if not math.isfinite(value) or value <= 0.0 or value > 1_000_000.0:
        return _RATE_MESSAGE
    return None


def _probe_errno(path: str) -> int | None:
    """Return the errno of a failed create, or ``None`` when a byte is durable.

    An empty create can succeed on a tmpfs that has no data blocks left, so
    the probe writes one byte and fsyncs it.
    """
    probe = os.path.join(path, f".dipcatcher-write-probe-{os.getpid()}")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(probe, flags, 0o644)
    except OSError as exc:
        return exc.errno if exc.errno is not None else errno.EIO
    failure = _write_probe(descriptor)
    os.close(descriptor)
    try:
        os.unlink(probe)
    except OSError:
        if failure is None:
            return errno.EIO
    return failure


def _write_probe(descriptor: int) -> int | None:
    try:
        os.write(descriptor, b"x")
        os.fsync(descriptor)
    except OSError as exc:
        return exc.errno if exc.errno is not None else errno.EIO
    return None


def _write_problem(path: str, failure: int) -> str:
    if failure == errno.ENOSPC:
        return f"data directory {path} is full; the dipcatcher user is uid {_DIP_UID}"
    euid = os.geteuid()
    return (
        f"data directory {path} is not writable by uid {euid}; "
        f"the dipcatcher user is uid {_DIP_UID}"
    )


def _research_command(argv: list[str]) -> bool:
    return any(_RESEARCH_ARGV in arg for arg in argv[1:])


def check_research_data_dir(path: str) -> str | None:
    """Return a message when production research can write ``path``."""
    if os.path.islink(path):
        return f"data directory {path} must not be a symlink; the dipcatcher user is uid {_DIP_UID}"
    if not os.path.isdir(path):
        return f"data directory {path} is not a directory; the dipcatcher user is uid {_DIP_UID}"
    failure = _probe_errno(path)
    if failure in _READ_ONLY_ERRNO:
        return None
    if failure == errno.ENOSPC:
        return _write_problem(path, failure)
    return f"research data directory {path} {_RESEARCH_RO}"


def check_data_dir(path: str, *, require_write: bool) -> str | None:
    """Return a message when ``path`` cannot be used, else ``None``."""
    if os.path.islink(path):
        return f"data directory {path} must not be a symlink; the dipcatcher user is uid {_DIP_UID}"
    if not os.path.isdir(path):
        return f"data directory {path} is not a directory; the dipcatcher user is uid {_DIP_UID}"
    if not require_write:
        return None
    failure = _probe_errno(path)
    if failure is None:
        return None
    return _write_problem(path, failure)


def _single_token(value: str) -> bool:
    if not value or any(char.isspace() for char in value):
        return False
    return not any(char in value for char in "\"'{}#")


def _read_secret(path: str) -> str | None:
    """Return the file text, or ``None`` when it cannot be read. Do not log it."""
    if not path or os.path.islink(path):
        return None
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size == 0 or info.st_size > _SECRET_MAX_BYTES:
            return None
        raw = os.read(descriptor, _SECRET_MAX_BYTES)
    except OSError:
        return None
    finally:
        os.close(descriptor)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if text.endswith("\n"):
        text = text[:-1]
    if text.endswith("\r"):
        text = text[:-1]
    return text


def _one_secret(name: str, production: bool) -> str | None:
    file_var = f"{name}_FILE"
    path = os.environ.get(file_var, "").strip()
    plain = os.environ.get(name, "").strip()
    if path and plain:
        return f"{name} must not be set in the environment when {file_var} is set"
    if production and plain:
        return f"{name} must be loaded from {file_var} when DIPCATCHER_PROFILE is production"
    if not path:
        return None
    value = _read_secret(path)
    if value is None or not _single_token(value):
        return f"{file_var} must contain one token"
    return None


def _webhook_url_ok(url: str) -> bool:
    if len(url) > 512 or any(char.isspace() for char in url):
        return False
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return False
    return not parts.username and not parts.password


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


def _secret_problems() -> list[str]:
    production = _production()
    problems = [problem for name in _SECRET_NAMES if (problem := _one_secret(name, production))]
    webhook = _webhook_problem(production)
    if webhook:
        problems.append(webhook)
    return problems


def main(argv: list[str]) -> int:
    problems: list[str] = []
    problems.extend(_secret_problems())
    _apply_production_rate_default()
    rate = _production_rate_problem()
    if rate is not None:
        problems.append(rate)
    root = os.environ.get("DIPCATCHER_DATA_ROOT", "/app/data")
    if _research_command(argv) and _production():
        message = check_research_data_dir(root)
    else:
        require_write = _enabled("DIPCATCHER_REQUIRE_DATA_WRITE", "0") or _production()
        message = check_data_dir(root, require_write=require_write)
    if message is not None:
        problems.append(message)
    if problems:
        sys.stderr.write("\n".join(problems) + "\n")
        return 1
    if len(argv) < 2:
        sys.stderr.write("entrypoint requires a command\n")
        return 1
    os.execvp(argv[1], argv[1:])
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
