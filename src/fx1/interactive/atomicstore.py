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
import errno
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
_MAX_JSON_BYTES = 16 * 1024 * 1024


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
    """Create *path* and any missing parents at mode ``0700`` on POSIX.

    A symlink at *path* or any existing ancestor is refused before a directory
    is created, so a planted link cannot redirect private files. An existing
    leaf is tightened. Ancestors that already existed are not chmod'd.
    """
    _refuse_symlink_chain(path)
    _make_private_dirs(path)
    _tighten_leaf(path)
    _refuse_symlink_chain(path)
    return path


def _chain(path: Path) -> tuple[Path, ...]:
    """Root-first components of *path* without constructing a new path class."""
    items: list[Path] = []
    current = path
    while True:
        items.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    items.reverse()
    return tuple(items)


def _refuse_symlink_chain(path: Path) -> None:
    for current in _chain(path):
        if _is_link(current):
            raise StoreError(f"refusing to follow symlink directory {path}")
        if not current.exists():
            return


def _is_link(path: Path) -> bool:
    try:
        return path.is_symlink()
    except OSError:
        return False


def _make_private_dirs(path: Path) -> None:
    for current in _chain(path):
        _ensure_dir(current, path)


def _ensure_dir(current: Path, original: Path) -> None:
    if _is_link(current):
        raise StoreError(f"refusing to follow symlink directory {original}")
    if current.exists():
        return
    _mkdir_private(current, original)


def _mkdir_private(path: Path, original: Path) -> None:
    try:
        os.mkdir(path, DIR_MODE)
    except FileExistsError:
        if _is_link(path):
            raise StoreError(f"refusing to follow symlink directory {original}") from None
        return
    if os.name == "posix":
        _fchmod_dir(path)


def _fchmod_dir(path: Path) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        os.fchmod(fd, DIR_MODE)
    finally:
        os.close(fd)


def _tighten_leaf(path: Path) -> None:
    if os.name != "posix" or path.parent == path:
        return
    if _is_link(path) or not path.is_dir():
        return
    with contextlib.suppress(OSError):
        _fchmod_dir(path)


def _fsync_dir(directory: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    with contextlib.suppress(OSError):
        fd = os.open(directory, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write_bytes(
    path: Path, data: bytes, *, mode: int = FILE_MODE, exclusive: bool = False
) -> None:
    """Replace *path* with *data*. ``exclusive`` fails if the name already exists.

    The previous content survives a crash. ``exclusive`` uses ``link`` so a
    lost exists-check cannot clobber a file, symlink, or FIFO that appeared.
    """
    ensure_private_dir(path.parent)
    tmp = _write_temp(path, data, mode)
    try:
        _publish(tmp, path, exclusive=exclusive)
        _fsync_dir(path.parent)
    except BaseException:
        _discard(tmp)
        raise


def _write_temp(path: Path, data: bytes, mode: int) -> Path:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        _write_all(fd, data)
        os.fsync(fd)
        if os.name == "posix":
            os.fchmod(fd, mode)
    except BaseException:
        os.close(fd)
        _discard(tmp)
        raise
    os.close(fd)
    return tmp


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while len(view) > 0:
        wrote = os.write(fd, view)
        if wrote <= 0:
            raise OSError("short write")
        view = view[wrote:]


def _publish(tmp: Path, path: Path, *, exclusive: bool) -> None:
    if not exclusive:
        os.replace(tmp, path)
        return
    try:
        os.link(tmp, path)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            raise FileExistsError(path) from exc
        raise
    _discard(tmp)


def _discard(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


def atomic_write_text(
    path: Path, text: str, *, mode: int = FILE_MODE, exclusive: bool = False
) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode, exclusive=exclusive)


def atomic_write_json(
    path: Path,
    obj: Any,
    *,
    mode: int = FILE_MODE,
    indent: int | None = 2,
    exclusive: bool = False,
) -> None:
    """Serialize *obj* deterministically (sorted keys, UTF-8) and write atomically."""
    text = json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False, allow_nan=False)
    atomic_write_text(path, text + "\n", mode=mode, exclusive=exclusive)


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
    Symlinks, hard links, FIFOs, devices, and files over 16 MiB are refused
    before their bytes are parsed.
    """
    try:
        raw = _read_private(path)
    except IsADirectoryError as exc:
        raise StoreCorruptError(path, "is a directory") from exc
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_reject_non_json)
    except (UnicodeDecodeError, ValueError, RecursionError, OverflowError) as exc:
        if quarantine_corrupt:
            quarantine(path)
        raise StoreCorruptError(path, f"not valid JSON ({type(exc).__name__})") from exc


def _read_private(path: Path) -> bytes:
    fd = _open_nofollow(path)
    try:
        return _read_private_fd(fd, path)
    finally:
        os.close(fd)


def _read_private_fd(fd: int, path: Path) -> bytes:
    info = os.fstat(fd)
    if stat.S_ISDIR(info.st_mode):
        raise IsADirectoryError(errno.EISDIR, "Is a directory", os.fspath(path))
    if not stat.S_ISREG(info.st_mode) or info.st_nlink > 1:
        raise OSError("refusing to read a non-private file")
    if info.st_size > _MAX_JSON_BYTES:
        raise OSError("file exceeds size limit")
    return _read_capped_fd(fd, _MAX_JSON_BYTES)


def _open_nofollow(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if os.name == "posix":
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    return os.open(path, flags)


def _read_capped_fd(fd: int, limit: int) -> bytes:
    chunks: list[bytes] = []
    remaining = limit
    while remaining > 0:
        block = os.read(fd, min(remaining, 1024 * 1024))
        if not block:
            break
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def _open_lock(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if os.name == "posix":
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags, FILE_MODE)
    try:
        _require_lock_fd(fd)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _require_lock_fd(fd: int) -> None:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink > 1:
        raise StoreError("lock file must be a private regular file")
    if os.name == "posix":
        os.fchmod(fd, FILE_MODE)


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
        fd = _open_lock(self._path)
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
