"""Single-writer gate for the paper ledger and the operator kill file.

``DIPCATCHER_WRITE_OWNER`` is not part of the service env contract. The two
replicas need different values, so compose sets them and the example env
file does not. Unset keeps the one-process behavior: writes are allowed and
no lock is taken.

``1``, ``true``, ``yes``, or ``on`` is the owner. :func:`claim_write_lock`
takes an exclusive non-blocking flock on
``{DIPCATCHER_DATA_ROOT}/.dipcatcher-write.lock`` and holds the descriptor
for the process lifetime. A second owner fails to boot.

``0``, ``false``, ``no``, or ``off`` is a reader. :func:`require_write_owner`
raises before a ledger or kill-file write. Readers still read those files.
An invalid value raises ``write_owner_invalid``.
"""

from __future__ import annotations

import fcntl
import os
from contextlib import suppress
from pathlib import Path

_ENV = "DIPCATCHER_WRITE_OWNER"
_OWNER = frozenset({"1", "true", "yes", "on"})
_READER = frozenset({"0", "false", "no", "off"})
_LOCK_NAME = ".dipcatcher-write.lock"
_HELD: list[int] = []


class WriteOwnerError(RuntimeError):
    """A replica that is not the writer tried to boot or to write."""


def write_owner_mode() -> str:
    """Return ``unset``, ``owner``, or ``reader``. An invalid value raises."""
    raw = os.environ.get(_ENV)
    if raw is None or not raw.strip():
        return "unset"
    token = raw.strip().lower()
    if token in _OWNER:
        return "owner"
    if token in _READER:
        return "reader"
    raise WriteOwnerError(f"write_owner_invalid: {_ENV} is invalid")


def _root(data_root: Path | None) -> Path:
    if data_root is not None:
        return data_root
    raw = os.environ.get("DIPCATCHER_DATA_ROOT", "data").strip() or "data"
    return Path(raw)


def _open_lock(root: Path) -> int:
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        return os.open(root / _LOCK_NAME, flags, 0o644)
    except OSError as exc:
        raise WriteOwnerError(f"write_owner_busy: {_ENV} could not open the lock") from exc


def claim_write_lock(data_root: Path | None = None) -> None:
    """Hold the data-directory lock when this process is the owner.

    Readers and an unset variable return without creating the lock file.
    """
    if write_owner_mode() != "owner":
        return
    root = _root(data_root)
    if root.is_symlink() or not root.is_dir():
        raise WriteOwnerError(f"write_owner_busy: {_ENV} data root is not a directory")
    descriptor = _open_lock(root)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(descriptor)
        raise WriteOwnerError(f"write_owner_busy: {_ENV} is already held") from exc
    _HELD.append(descriptor)


def _deny_message(mode: str) -> str:
    if mode == "reader":
        return f"write_owner_required: {_ENV} is not the writer"
    return f"write_owner_required: {_ENV} has not claimed the lock"


def require_write_owner() -> None:
    """Allow a write for unset and for an owner that holds the lock."""
    mode = write_owner_mode()
    if mode == "reader" or (mode == "owner" and not _HELD):
        raise WriteOwnerError(_deny_message(mode))


def release_write_lock() -> None:
    """Drop a lock held in this process. Used by tests."""
    while _HELD:
        descriptor = _HELD.pop()
        with suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        with suppress(OSError):
            os.close(descriptor)
