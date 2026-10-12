"""Atomic, private, lock-protected persistence primitives.

Everything that outlives a process — credentials, sessions, job ledgers,
history — is written through here so a crash, a full disk, or a second
concurrent ``dipcatcher`` process can never leave a half-written file behind:

* ``atomic_write_bytes`` writes a sibling temp file, fsyncs it, and
  ``os.replace``s it over the target (the old content survives any failure).
* Files are created ``0600`` and directories ``0700`` on POSIX.
* ``FileLock`` serializes read-modify-write cycles across threads and
  processes (``fcntl`` on POSIX, ``msvcrt`` on Windows).
* ``read_json`` reports corruption explicitly and can quarantine the bad file
  instead of silently discarding evidence.
"""

from __future__ import annotations

import contextlib
import json
import os
import stat
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any

FILE_MODE = 0o600
DIR_MODE = 0o700


class StoreError(RuntimeError):
    """Base class for persistence failures."""


class StoreCorruptError(StoreError):
    """A stored file exists but is not valid JSON."""

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"{path.name}: {reason}")
        self.path = path
        self.reason = reason


class StoreLockTimeout(StoreError):
    """The advisory lock could not be acquired in time."""


def ensure_private_dir(path: Path) -> Path:
    """Create *path* (and parents) and make that directory ``0700`` on POSIX.

    An existing directory is tightened too. A symlink is refused so a planted
    link cannot redirect private files.
    """
    if path.is_symlink():
        raise StoreError(f"refusing to follow symlink directory {path}")
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        with contextlib.suppress(OSError):
            os.chmod(path, DIR_MODE)
    return path


def _fsync_dir(directory: Path) -> None:
    if os.name != "posix":
        return
    with contextlib.suppress(OSError):
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write_bytes(path: Path, data: bytes, *, mode: int = FILE_MODE) -> None:
    """Replace *path* with *data* atomically; the previous content survives failure."""
    ensure_private_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if os.name == "posix":
            os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
        _fsync_dir(path.parent)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def atomic_write_text(path: Path, text: str, *, mode: int = FILE_MODE) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def atomic_write_json(
    path: Path, obj: Any, *, mode: int = FILE_MODE, indent: int | None = 2
) -> None:
    """Serialize *obj* deterministically (sorted keys, UTF-8) and write atomically."""
    text = json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False, allow_nan=False)
    atomic_write_text(path, text + "\n", mode=mode)


def quarantine(path: Path) -> Path | None:
    """Move a bad file aside; ``None`` if it vanished.

    The suffix includes pid and a monotonic tick so two quarantines in the
    same wall-clock second cannot overwrite each other.
    """
    token = f"{int(time.time())}-{os.getpid()}-{time.monotonic_ns()}"
    target = path.with_name(f"{path.name}.corrupt-{token}")
    try:
        os.replace(path, target)
    except OSError:
        return None
    return target


def read_json(path: Path, *, quarantine_corrupt: bool = False) -> Any:
    """Load JSON from *path*.

    Raises ``FileNotFoundError`` when absent and ``StoreCorruptError`` when the
    content is not valid UTF-8 JSON (optionally moving the file aside first).
    """
    try:
        raw = path.read_bytes()
    except IsADirectoryError as exc:
        raise StoreCorruptError(path, "is a directory") from exc
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_reject_non_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        if quarantine_corrupt:
            quarantine(path)
        raise StoreCorruptError(path, f"not valid JSON ({type(exc).__name__})") from exc


def _reject_non_json(value: str) -> Any:
    """``NaN`` / ``Infinity`` are not JSON; surface them as corruption."""
    raise json.JSONDecodeError(f"invalid JSON constant {value}", value, 0)


def permission_problem(path: Path) -> str | None:
    """Why *path* is readable by others (POSIX), else ``None``.

    Windows ACLs are not inspected here; callers that need them (credential
    storage) apply platform-specific protection themselves.
    """
    if os.name != "posix":
        return None
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return None
    if mode & 0o077:
        return f"{path} is group/world accessible (mode {mode:04o}); run: chmod 600 {path}"
    return None


class FileLock:
    """Advisory exclusive lock on ``<path>.lock`` (threads and processes)."""

    def __init__(self, path: Path, *, timeout_s: float = 5.0, poll_s: float = 0.02) -> None:
        self._path = path.with_name(path.name + ".lock")
        self._timeout_s = timeout_s
        self._poll_s = poll_s
        self._fd: int | None = None

    def _prepare_windows_lock(self, fd: int) -> None:
        # msvcrt locks a byte range and fails when that range is past EOF.
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            if os.fstat(fd).st_size < 1:
                os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
        except OSError:
            return

    def _try_lock(self, fd: int) -> bool:
        if sys.platform == "win32":
            import msvcrt

            self._prepare_windows_lock(fd)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError:
                return False
            return True
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _unlock(self, fd: int) -> None:
        if sys.platform == "win32":
            import msvcrt

            with contextlib.suppress(OSError):
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)

    def __enter__(self) -> FileLock:
        ensure_private_dir(self._path.parent)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, FILE_MODE)
        deadline = time.monotonic() + self._timeout_s
        while not self._try_lock(fd):
            if time.monotonic() >= deadline:
                os.close(fd)
                raise StoreLockTimeout(
                    f"could not lock {self._path.name} within {self._timeout_s}s"
                )
            time.sleep(self._poll_s)
        self._fd = fd
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fd is not None:
            self._unlock(self._fd)
            os.close(self._fd)
            self._fd = None


def update_json(
    path: Path,
    mutate: Callable[[Any], Any],
    *,
    default: Any = None,
    lock_timeout_s: float = 5.0,
) -> Any:
    """Locked read-modify-write: ``mutate(current) -> new`` is persisted atomically.

    A missing file starts from ``default`` (deep-copied through JSON so callers
    cannot alias it). A corrupt file raises ``StoreCorruptError`` rather than
    being overwritten.
    """
    with FileLock(path, timeout_s=lock_timeout_s):
        try:
            current = read_json(path)
        except FileNotFoundError:
            current = json.loads(json.dumps(default))
        updated = mutate(current)
        atomic_write_json(path, updated)
        return updated
