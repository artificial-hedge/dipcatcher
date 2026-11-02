"""Fail-closed path containment and private file creates.

Config paths must stay inside their root. ``..``, an absolute spelling, and
symbolic links are refused, so a config value cannot point a later read or
write at another directory. Durable outputs are created mode ``0o600`` and
are never written through a symlink.
"""

from __future__ import annotations

import errno
import os
import tempfile
from pathlib import Path

_PRIVATE_MODE = 0o600


class PathSafetyError(ValueError):
    """A path escaped its root or was not a regular file we can own."""


def _nofollow_flags(flags: int) -> int:
    return flags | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)


def _reject_control_chars(raw: str) -> None:
    if not isinstance(raw, str) or raw.strip() == "" or any(ord(char) < 32 for char in raw):
        raise PathSafetyError("path must be a non-empty string without control characters")


def _refuse_symlink(path: Path) -> None:
    """Refuse a symlink or an existing non-file before any create or replace."""
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise PathSafetyError(f"refusing to write through an unsafe path: {path}")


def _reraise_open(path: Path, exc: OSError) -> None:
    loop = getattr(errno, "ELOOP", None)
    if path.is_symlink() or (loop is not None and exc.errno == loop):
        raise PathSafetyError(f"refusing to write through an unsafe path: {path}") from exc


def _owned(fd: int) -> int:
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, _PRIVATE_MODE)
    except OSError:
        os.close(fd)
        raise
    return fd


def contained_path(root: Path, raw: str) -> Path:
    """Resolve ``raw`` and require every component to stay inside ``root``.

    The result is absolute. A symlink anywhere in the relative path raises,
    including a link whose target happens to stay inside ``root``.
    """
    _reject_control_chars(raw)
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PathSafetyError(f"path escapes its base directory: {raw}")
    base = Path(root).resolve()
    probe = base
    for part in candidate.parts:
        if part in {"", ".", ".."}:
            raise PathSafetyError(f"path escapes its base directory: {raw}")
        probe = probe / part
        if probe.is_symlink():
            raise PathSafetyError(f"path contains a symlink: {raw}")
    resolved = probe.resolve()
    if not resolved.is_relative_to(base):
        raise PathSafetyError(f"path escapes its base directory: {raw}")
    return resolved


def output_path(raw: str, *roots: Path) -> Path:
    """Return ``raw`` when its resolved path stays inside one of ``roots``.

    Relative paths resolve from the current directory. The caller's spelling
    is preserved. Parent segments and symbolic links are refused.
    """
    if not roots:
        raise PathSafetyError("output path needs a permitted directory")
    _reject_control_chars(raw)
    candidate = Path(raw)
    if ".." in candidate.parts or candidate.is_symlink():
        raise PathSafetyError(f"path escapes its permitted directory: {raw}")
    resolved = candidate.resolve()
    allowed = tuple(Path(root).resolve() for root in roots)
    if not any(resolved.is_relative_to(base) for base in allowed):
        raise PathSafetyError(f"path escapes its permitted directory: {raw}")
    return candidate


def open_private(path: Path, flags: int) -> int:
    """Open ``path`` as mode ``0o600``. A symlink or non-file raises.

    The caller closes the descriptor. ``O_NOFOLLOW`` is set when the platform
    provides it, so a symlink swapped in after the check is not followed.
    """
    destination = Path(path)
    _refuse_symlink(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(destination, _nofollow_flags(flags), _PRIVATE_MODE)
    except OSError as exc:
        _reraise_open(destination, exc)
        raise
    return _owned(fd)


def write_private_text(path: Path, text: str) -> Path:
    """Replace ``path`` with UTF-8 text stored as mode ``0o600``.

    An existing symlink is left in place. The bytes land in a same-directory
    temporary file and then ``os.replace`` publishes them.
    """
    destination = Path(path)
    _refuse_symlink(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            if hasattr(os, "fchmod"):
                os.fchmod(handle.fileno(), _PRIVATE_MODE)
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        _refuse_symlink(destination)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination
