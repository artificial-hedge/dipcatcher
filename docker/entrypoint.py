"""Fail fast before the API process is executed.

A symlink is always refused. The write probe runs when
``DIPCATCHER_REQUIRE_DATA_WRITE`` is on, and also when
``DIPCATCHER_PROFILE`` is ``production`` or ``prod``, unless the command is
the research API. Production research must see a data directory it cannot
write. That profile also refuses to boot unless ``DIPCATCHER_RATE_LIMIT_RPS``
is set and above 0. The research image leaves the write flag off;
``deploy/prod.env`` turns the profile on. This file uses the stdlib only.
"""

from __future__ import annotations

import errno
import math
import os
import sys

_DIP_UID = 10001
_TRUE = frozenset({"1", "true", "yes", "on"})
_PRODUCTION = frozenset({"production", "prod"})
_RATE_MESSAGE = "DIPCATCHER_RATE_LIMIT_RPS must be set above 0 in production"
_RESEARCH_ARGV = "quant_fund.api.research_api"
_READ_ONLY_ERRNO = frozenset({errno.EROFS, errno.EACCES, errno.EPERM})
_RESEARCH_RO = "must be read-only when DIPCATCHER_PROFILE is production"


def _enabled(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUE


def _production() -> bool:
    return os.environ.get("DIPCATCHER_PROFILE", "").strip().lower() in _PRODUCTION


def _production_rate_problem() -> str | None:
    """Return the production rate message, or ``None`` when the boot may continue."""
    if not _production():
        return None
    raw = os.environ.get("DIPCATCHER_RATE_LIMIT_RPS", "").strip()
    if not raw:
        return _RATE_MESSAGE
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


def main(argv: list[str]) -> int:
    problems: list[str] = []
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
