"""Fail fast when the lab data directory is not writable by uid 10001.

The research image sets ``DIPCATCHER_REQUIRE_DATA_WRITE=0`` so a read-only
mount still starts. A symlink is refused in both modes. This file uses the
stdlib only.
"""

from __future__ import annotations

import errno
import os
import sys

_DIP_UID = 10001
_TRUE = frozenset({"1", "true", "yes", "on"})


def _enabled(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUE


def _probe_errno(path: str) -> int | None:
    """Return the errno of a failed create, or ``None`` when the probe lands."""
    probe = os.path.join(path, f".dipcatcher-write-probe-{os.getpid()}")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(probe, flags, 0o644)
    except OSError as exc:
        return exc.errno if exc.errno is not None else errno.EIO
    os.close(descriptor)
    os.unlink(probe)
    return None


def _write_problem(path: str, failure: int) -> str:
    if failure == errno.ENOSPC:
        return f"data directory {path} is full; the dipcatcher user is uid {_DIP_UID}"
    euid = os.geteuid()
    return (
        f"data directory {path} is not writable by uid {euid}; "
        f"the dipcatcher user is uid {_DIP_UID}"
    )


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
    root = os.environ.get("DIPCATCHER_DATA_ROOT", "/app/data")
    message = check_data_dir(root, require_write=_enabled("DIPCATCHER_REQUIRE_DATA_WRITE", "0"))
    if message is not None:
        sys.stderr.write(message + "\n")
        return 1
    if len(argv) < 2:
        sys.stderr.write("entrypoint requires a command\n")
        return 1
    os.execvp(argv[1], argv[1:])
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
