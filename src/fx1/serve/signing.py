"""Release signing for fx-1 checkpoints (attestation ladder, tier 1).

A release signs the complete regular-file inventory beneath a checkpoint,
except its two root-level signature metadata files. Artifacts are hashed
incrementally, so verification does not allocate an entire weights file.
The signing key comes from ``FX1_SIGNING_KEY`` only.

This is detached HMAC verification, not a Sigstore signature or a TEE
attestation. The checkpoint must not be mutated concurrently with signing,
verification, or loading; this module does not lock the deployment tree.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, cast

from pydantic import BaseModel, Field

SIGNING_KEY_ENV = "FX1_SIGNING_KEY"
SIGNATURE_FILENAME = "release.sig"
MANIFEST_FILENAME = "release.manifest.json"

_METADATA_FILENAMES = frozenset({SIGNATURE_FILENAME, MANIFEST_FILENAME})
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_SIGNATURE_BYTES = 4096


class ReleaseManifest(BaseModel):
    """Digest manifest of everything a release consists of."""

    checkpoint_dir: str
    artifacts: dict[str, str] = Field(min_length=1, description="relative path -> sha256")


@contextmanager
def _open_regular_file(path: Path) -> Iterator[BinaryIO]:
    """Open a regular file without following a final-component symlink."""
    if path.is_symlink():
        raise ValueError(f"release files must not be symlinks: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"release files must be regular files: {path}")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            yield stream
    finally:
        os.close(fd)


def _hash_file(path: Path) -> str:
    with _open_regular_file(path) as stream:
        # ``os.fdopen(..., "rb")`` supplies ``readinto`` at runtime, but the
        # broad BinaryIO protocol in typeshed does not expose that member.
        return hashlib.file_digest(cast(Any, stream), "sha256").hexdigest()


def _raise_walk_error(error: OSError) -> None:
    raise error


def _artifact_paths(root: Path) -> Iterator[Path]:
    """Enumerate the same closed inventory for both signing and verification."""
    for directory, directories, filenames in os.walk(root, onerror=_raise_walk_error):
        parent = Path(directory)
        for name in directories:
            if (parent / name).is_symlink():
                raise ValueError(f"release directories must not be symlinks: {parent / name}")
        for name in filenames:
            path = parent / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError(f"release files must be regular files: {path}")
            if parent == root and name in _METADATA_FILENAMES:
                continue
            yield path


def build_manifest(checkpoint_dir: str | Path) -> ReleaseManifest:
    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found: {root}")
    artifacts = {
        str(path.relative_to(root)): _hash_file(path) for path in sorted(_artifact_paths(root))
    }
    if not artifacts:
        raise ValueError(f"checkpoint dir {root} contains no artifacts")
    return ReleaseManifest(checkpoint_dir=str(root), artifacts=artifacts)


def _key() -> bytes:
    key = os.environ.get(SIGNING_KEY_ENV, "")
    if not key:
        raise RuntimeError(f"{SIGNING_KEY_ENV} is not set; fx-1 never hardcodes signing keys")
    return key.encode()


def _read_metadata(path: Path, limit: int) -> bytes:
    with _open_regular_file(path) as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError(f"release metadata exceeds its byte limit: {path}")
    return content


def _write_regular_file(path: Path, data: bytes) -> None:
    """Write release metadata without following a final-component symlink.

    ``Path.write_bytes`` resolves a pre-planted ``release.sig`` symlink and
    clobbers whatever it points at (e.g. a user's config outside the
    checkpoint); ``O_NOFOLLOW`` refuses instead, matching the read invariant.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o666)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"release metadata must not be a symlink: {path}") from exc
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"release metadata must be a regular file: {path}")
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(data)
    finally:
        os.close(fd)


def sign_release(checkpoint_dir: str | Path) -> Path:
    """Write a manifest and detached HMAC; validate configuration before I/O."""
    key = _key()
    root = Path(checkpoint_dir)
    manifest_bytes = build_manifest(root).model_dump_json().encode()
    if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
        raise ValueError("release manifest exceeds its byte limit")
    signature = hmac.new(key, manifest_bytes, hashlib.sha256).hexdigest()
    _write_regular_file(root / MANIFEST_FILENAME, manifest_bytes)
    sig_path = root / SIGNATURE_FILENAME
    _write_regular_file(sig_path, signature.encode("ascii"))
    return sig_path


def verify_release(checkpoint_dir: str | Path) -> bool:
    """Verify the HMAC and exact inventory; malformed/unreadable releases fail closed.

    Missing signing configuration still raises ``RuntimeError`` when release
    metadata exists. Authentication precedes artifact hashing. Manifest paths
    are never opened: they are compared with paths enumerated beneath root.
    """
    root = Path(checkpoint_dir)
    manifest_path = root / MANIFEST_FILENAME
    sig_path = root / SIGNATURE_FILENAME
    try:
        if not manifest_path.exists() or not sig_path.exists():
            return False
        manifest_bytes = _read_metadata(manifest_path, _MAX_MANIFEST_BYTES)
        expected = hmac.new(_key(), manifest_bytes, hashlib.sha256).hexdigest().encode("ascii")
        signature = _read_metadata(sig_path, _MAX_SIGNATURE_BYTES).strip()
        if not hmac.compare_digest(expected, signature):
            return False
        manifest = ReleaseManifest.model_validate_json(manifest_bytes)
        paths = {str(path.relative_to(root)): path for path in _artifact_paths(root)}
        if paths.keys() != manifest.artifacts.keys():
            return False
        # Reject cheap inventory mismatches before reading potentially huge weights.
        # Hash every matching artifact afresh; metadata is not a digest cache.
        return all(_hash_file(path) == manifest.artifacts[name] for name, path in paths.items())
    except (OSError, ValueError):
        return False
