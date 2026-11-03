"""Halt new orders when a durable write cannot land.

A full disk or a read-only directory sets a process latch. The latch is not
cleared until a later probe writes a byte. Callers must not swallow the
failure and report the order as accepted.
"""

from __future__ import annotations

import errno
import os
import time
from pathlib import Path

_REASON: str | None = None
_HOLD_SKIPS = 0
_CODES = frozenset({"enospc", "read_only", "write_failed"})
_PATHS = (
    "DIPCATCHER_LEDGER_PATH",
    "DIPCATCHER_JOURNAL_PATH",
    "DIPCATCHER_RATE_STORE_PATH",
    "DIPCATCHER_AUDIT_LOG_PATH",
)


class DurableWriteError(OSError):
    """A ledger, journal, rate-store, or audit write did not land."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def write_halt_reason() -> str | None:
    """The latch code, or ``None`` when durable writes are currently landing."""
    return _REASON


def note_write_failure(reason: str) -> None:
    """Latch new orders. Unknown codes collapse to ``write_failed``."""
    global _REASON
    _REASON = reason if reason in _CODES else "write_failed"


def clear_write_halt() -> None:
    """Drop the latch after a probe write has landed."""
    global _REASON
    _REASON = None


def errno_code(err: int) -> str:
    """Map a filesystem errno to a stable halt code."""
    if err == errno.ENOSPC:
        return "enospc"
    if err in {errno.EROFS, errno.EACCES, errno.EPERM}:
        return "read_only"
    return "write_failed"


def configured_parents(extra: list[Path]) -> list[Path]:
    """Parents of the configured durable files, then ``extra``."""
    found: list[Path] = []
    for name in _PATHS:
        raw = os.environ.get(name, "").strip()
        if raw:
            found.append(Path(raw).parent)
    found.extend(extra)
    return found


def probe_directory(parent: Path) -> str | None:
    """Append one byte in ``parent``. A code means the byte did not land."""
    if parent.is_symlink() or not parent.is_dir():
        return "write_failed"
    probe = parent / ".dipcatcher-write-probe"
    if probe.is_symlink():
        return "write_failed"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(probe, flags, 0o644)
    except OSError as exc:
        return errno_code(exc.errno or 0)
    try:
        os.write(descriptor, b"\n")
        os.fsync(descriptor)
    except OSError as exc:
        return errno_code(exc.errno or 0)
    finally:
        os.close(descriptor)
    return None


def ensure_directories(parents: list[Path]) -> str | None:
    """Probe each parent. The first failure latches; success clears the latch."""
    if not parents:
        return write_halt_reason()
    for parent in parents:
        code = probe_directory(parent)
        if code:
            note_write_failure(code)
            return code
    clear_write_halt()
    return None


def _pause_before_replace() -> None:
    """Spin while ``DIPCATCHER_REPLACE_HOLD`` exists. Unset in production.

    ``DIPCATCHER_REPLACE_HOLD_SKIP`` counts publishes to let through first
    (the opening mark). The drill kills the process after the temporary
    bytes are durable and before ``os.replace`` publishes them.
    """
    global _HOLD_SKIPS
    raw = os.environ.get("DIPCATCHER_REPLACE_HOLD", "").strip()
    if not raw:
        return
    skip_raw = os.environ.get("DIPCATCHER_REPLACE_HOLD_SKIP", "").strip() or "0"
    try:
        skip = int(skip_raw)
    except ValueError:
        skip = 0
    if skip > _HOLD_SKIPS:
        _HOLD_SKIPS += 1
        return
    marker = Path(raw)
    if marker.is_symlink():
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("holding\n", encoding="utf-8")
    while marker.exists():
        time.sleep(0.02)


def replace_text(path: Path, payload: str) -> None:
    """Atomically replace ``path``. A failed write latches and raises."""
    parent = path.parent
    if parent.is_symlink():
        note_write_failure("write_failed")
        raise DurableWriteError("write_failed")
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{path.name}.tmp"
    if temporary.is_symlink():
        note_write_failure("write_failed")
        raise DurableWriteError("write_failed")
    try:
        temporary.write_text(payload, encoding="utf-8")
        _pause_before_replace()
        os.replace(temporary, path)
    except OSError as exc:
        code = errno_code(exc.errno or 0)
        note_write_failure(code)
        try:
            if not temporary.is_symlink() and temporary.exists():
                temporary.unlink()
        except OSError:
            pass
        raise DurableWriteError(code) from exc
