"""Backup and restore the paper ledger and checkpoint files.

The archive is ``manifest.json`` plus payload bytes. It includes the paper
ledger, checkpoints, and a regular ``kill_switch`` file at the source root
when that file exists. Restore checks every hash before it writes. A
symlink, a tampered payload, or a destination file with different bytes
fails closed and leaves the destination unchanged.
Copies land through a same-directory temporary file, fsync, and replace.
Dotfiles, including a torn write, are not archived.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath

from quant_fund.utils import atomicio
from quant_fund.utils.atomicio import atomic_write_text

SCHEMA = "paper_state_backup.v1"
_CHUNK = 1 << 20


class PaperBackupError(ValueError):
    """Backup or restore refused. The message is a code, not file contents."""


def backup_paper_state(source: Path, dest: Path) -> None:
    """Write an archive of the paper ledger and checkpoints under ``source``."""
    if source.is_symlink():
        raise PaperBackupError("symlink")
    if not source.is_dir():
        raise PaperBackupError("paper_state_missing")
    if dest.exists() or dest.is_symlink():
        raise PaperBackupError("dest_exists")
    files = _collect(source)
    dest.mkdir()
    payload_root = dest / "payload"
    payload_root.mkdir()
    entries: list[dict[str, object]] = []
    for path in files:
        relative = path.relative_to(source).as_posix()
        _safe_relative(relative)
        digest, size = _hash_file(path)
        target = payload_root.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_exclusive(path, target)
        entries.append({"path": relative, "sha256": digest, "size": size})
    manifest = {"files": entries, "schema": SCHEMA}
    atomic_write_text(
        dest / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )


def restore_paper_state(archive: Path, dest: Path) -> None:
    """Verify ``archive`` and then copy it onto ``dest``."""
    if archive.is_symlink() or dest.is_symlink():
        raise PaperBackupError("symlink")
    if not archive.is_dir():
        raise PaperBackupError("schema")
    entries = _manifest_entries(archive / "manifest.json")
    staged: list[tuple[str, Path]] = []
    for entry in entries:
        relative, digest, size = _entry_fields(entry)
        payload = archive / "payload" / Path(*PurePosixPath(relative).parts)
        _assert_file(payload)
        found_digest, found_size = _hash_file(payload)
        if found_digest != digest or found_size != size:
            raise PaperBackupError("hash_mismatch")
        _assert_dest_compatible(dest, relative, digest, size)
        staged.append((relative, payload))
    for relative, payload in staged:
        target = dest.joinpath(*PurePosixPath(relative).parts)
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_exclusive(payload, target)


def _collect(source: Path) -> list[Path]:
    roots = _select_roots(source)
    found: list[Path] = []
    for root in roots:
        found.extend(_walk_files(root))
    _include_kill(source, found)
    found.sort()
    return found


def _include_kill(source: Path, found: list[Path]) -> None:
    """Archive the operator kill file. A missing file stays ENABLED."""
    kill = source / "kill_switch"
    if kill.is_symlink():
        raise PaperBackupError("symlink")
    if not kill.exists():
        return
    if not kill.is_file():
        raise PaperBackupError("kill_switch")
    if kill not in found:
        found.append(kill)


def _select_roots(source: Path) -> list[Path]:
    metadata = source / "metadata"
    _assert_not_symlink(metadata)
    paper = metadata / "paper"
    checkpoints = source / "checkpoints"
    _assert_not_symlink(paper)
    _assert_not_symlink(checkpoints)
    roots = [path for path in (paper, checkpoints) if path.is_dir()]
    if roots:
        return roots
    ledger = source / "broker_state.json"
    _assert_not_symlink(ledger)
    if ledger.is_file() and not metadata.exists():
        return [source]
    raise PaperBackupError("paper_state_missing")


def _walk_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        _assert_not_symlink(current)
        kept: list[str] = []
        for name in dirnames:
            child = current / name
            _assert_not_symlink(child)
            if not name.startswith("."):
                kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            child = current / name
            _assert_not_symlink(child)
            if name.startswith(".") or not child.is_file():
                continue
            found.append(child)
    return found


def _assert_not_symlink(path: Path) -> None:
    if path.is_symlink():
        raise PaperBackupError("symlink")


def _assert_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise PaperBackupError("hash_mismatch")


def _safe_relative(relative: str) -> None:
    if not relative or relative.startswith(("/", "\\")) or "\\" in relative:
        raise PaperBackupError("path")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise PaperBackupError("path")


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _write_all(fd: int, chunk: bytes) -> None:
    view = memoryview(chunk)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError(errno.EIO, "short write")
        view = view[written:]


def _stream_to(fd: int, source: Path) -> None:
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                return
            _write_all(fd, chunk)


def _copy_exclusive(source: Path, target: Path) -> None:
    """Publish ``target`` only after the bytes are fsynced."""
    if target.exists() or target.is_symlink():
        raise PaperBackupError("dest_exists")
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary_path = Path(temporary)
    try:
        _stream_to(fd, source)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temporary_path, target)
        atomicio._fsync_directory(target.parent)
    finally:
        if fd >= 0:
            os.close(fd)
        temporary_path.unlink(missing_ok=True)


def _manifest_entries(path: Path) -> list[object]:
    if path.is_symlink() or not path.is_file():
        raise PaperBackupError("schema")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise PaperBackupError("schema") from None
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise PaperBackupError("schema")
    files = payload.get("files")
    if not isinstance(files, list):
        raise PaperBackupError("schema")
    return list(files)


def _entry_fields(entry: object) -> tuple[str, str, int]:
    if not isinstance(entry, dict):
        raise PaperBackupError("schema")
    relative = entry.get("path")
    digest = entry.get("sha256")
    size = entry.get("size")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise PaperBackupError("schema")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise PaperBackupError("schema")
    _safe_relative(relative)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise PaperBackupError("hash_mismatch")
    return relative, digest, size


def _assert_dest_compatible(dest: Path, relative: str, digest: str, size: int) -> None:
    target = dest.joinpath(*PurePosixPath(relative).parts)
    if target.is_symlink():
        raise PaperBackupError("symlink")
    if not target.exists():
        return
    if not target.is_file():
        raise PaperBackupError("dest_differs")
    found_digest, found_size = _hash_file(target)
    if found_digest != digest or found_size != size:
        raise PaperBackupError("dest_differs")
