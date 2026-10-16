"""Release signing for fx-1 checkpoints (attestation ladder, tier 1).

A release signs the complete regular-file inventory beneath a checkpoint,
except its two root-level signature metadata files. Artifacts are hashed
incrementally, so verification does not allocate an entire weights file.
Internal HMAC signing uses ``FX1_SIGNING_KEY``. Public releases use a detached
Ed25519 signature created with ``FX1_RELEASE_SIGNING_KEY`` on the build host;
inference needs only the trusted ``FX1_RELEASE_VERIFY_KEY`` public key.

These are detached signatures, not Sigstore signatures or TEE attestations.
The checkpoint must not be mutated concurrently with signing, verification,
or loading; this module does not lock the deployment tree.
"""

from __future__ import annotations

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
PUBLIC_SIGNING_KEY_ENV = "FX1_RELEASE_SIGNING_KEY"
PUBLIC_VERIFY_KEY_ENV = "FX1_RELEASE_VERIFY_KEY"
SIGNATURE_FILENAME = "release.sig"
PUBLIC_SIGNATURE_FILENAME = "release.ed25519.sig"
MANIFEST_FILENAME = "release.manifest.json"

_METADATA_FILENAMES = frozenset({SIGNATURE_FILENAME, PUBLIC_SIGNATURE_FILENAME, MANIFEST_FILENAME})
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


def sign_release(checkpoint_dir: str | Path) -> Path:
    """Write a manifest and detached HMAC; validate configuration before I/O."""
    key = _key()
    root = Path(checkpoint_dir)
    manifest_bytes = build_manifest(root).model_dump_json().encode()
    if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
        raise ValueError("release manifest exceeds its byte limit")
    signature = hmac.new(key, manifest_bytes, hashlib.sha256).hexdigest()
    (root / MANIFEST_FILENAME).write_bytes(manifest_bytes)
    sig_path = root / SIGNATURE_FILENAME
    sig_path.write_text(signature, encoding="utf-8")
    return sig_path


def sign_release_ed25519(checkpoint_dir: str | Path) -> Path:
    """Sign the exact checkpoint inventory with a build-host private seed."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    seed_hex = os.environ.get(PUBLIC_SIGNING_KEY_ENV, "").strip()
    if not seed_hex:
        raise RuntimeError(f"{PUBLIC_SIGNING_KEY_ENV} is not set")
    try:
        seed = bytes.fromhex(seed_hex)
        key = Ed25519PrivateKey.from_private_bytes(seed)
    except ValueError as exc:
        raise ValueError(f"{PUBLIC_SIGNING_KEY_ENV} must be a 32-byte hex seed") from exc
    root = Path(checkpoint_dir)
    manifest_bytes = build_manifest(root).model_dump_json().encode()
    if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
        raise ValueError("release manifest exceeds its byte limit")
    signature = key.sign(manifest_bytes)
    (root / MANIFEST_FILENAME).write_bytes(manifest_bytes)
    sig_path = root / PUBLIC_SIGNATURE_FILENAME
    sig_path.write_text(signature.hex(), encoding="ascii")
    return sig_path


def _verify_inventory(root: Path, manifest_bytes: bytes) -> bool:
    manifest = ReleaseManifest.model_validate_json(manifest_bytes)
    paths = {str(path.relative_to(root)): path for path in _artifact_paths(root)}
    if paths.keys() != manifest.artifacts.keys():
        return False
    # Reject cheap inventory mismatches before reading potentially huge weights.
    return all(_hash_file(path) == manifest.artifacts[name] for name, path in paths.items())


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
        return _verify_inventory(root, manifest_bytes)
    except (OSError, ValueError):
        return False


def verify_release_ed25519(checkpoint_dir: str | Path) -> bool:
    """Verify a public release using only a trusted public key and artifact bytes."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    root = Path(checkpoint_dir)
    manifest_path = root / MANIFEST_FILENAME
    sig_path = root / PUBLIC_SIGNATURE_FILENAME
    if not manifest_path.exists() or not sig_path.exists():
        return False
    public_hex = os.environ.get(PUBLIC_VERIFY_KEY_ENV, "").strip()
    if not public_hex:
        raise RuntimeError(f"{PUBLIC_VERIFY_KEY_ENV} is not set")
    try:
        public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_hex))
        manifest_bytes = _read_metadata(manifest_path, _MAX_MANIFEST_BYTES)
        signature = bytes.fromhex(_read_metadata(sig_path, _MAX_SIGNATURE_BYTES).decode("ascii"))
        public_key.verify(signature, manifest_bytes)
        return _verify_inventory(root, manifest_bytes)
    except (InvalidSignature, OSError, UnicodeError, ValueError):
        return False
