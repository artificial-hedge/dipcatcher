"""SYNTHETIC adversarial probes for fx1.serve.signing — key handling,
attestation inventory, and metadata hygiene."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fx1.serve.signing import (
    MANIFEST_FILENAME,
    SIGNATURE_FILENAME,
    SIGNING_KEY_ENV,
    build_manifest,
    sign_release,
    verify_release,
)


def _checkpoint(root: Path) -> Path:
    (root / "weights.bin").write_bytes(b"SYNTHETIC weights")
    (root / "config.json").write_bytes(b'{"model": "SYNTHETIC"}')
    return root


def test_sign_and_verify_leave_the_env_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The signing key is read per call — neither signing nor verification
    may mutate the process env (an opt-out must not leak past its scope)."""
    monkeypatch.setenv(SIGNING_KEY_ENV, "synthetic-signing-key")
    _checkpoint(tmp_path)
    before = dict(os.environ)
    sign_release(tmp_path)
    assert verify_release(tmp_path) is True
    assert dict(os.environ) == before


def test_resign_is_deterministic_and_still_verifies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(SIGNING_KEY_ENV, "synthetic-signing-key")
    _checkpoint(tmp_path)
    sign_release(tmp_path)
    first_manifest = (tmp_path / MANIFEST_FILENAME).read_bytes()
    first_sig = (tmp_path / SIGNATURE_FILENAME).read_bytes()
    sign_release(tmp_path)
    assert (tmp_path / MANIFEST_FILENAME).read_bytes() == first_manifest
    assert (tmp_path / SIGNATURE_FILENAME).read_bytes() == first_sig
    assert verify_release(tmp_path) is True


def test_verify_under_a_different_key_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(SIGNING_KEY_ENV, "key-a")
    _checkpoint(tmp_path)
    sign_release(tmp_path)
    monkeypatch.setenv(SIGNING_KEY_ENV, "key-b")
    assert verify_release(tmp_path) is False


def test_unsigned_dir_verifies_false_and_missing_key_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No metadata -> False (not an exception); metadata present but the
    key unset -> RuntimeError (configuration fault, not a bad release)."""
    monkeypatch.delenv(SIGNING_KEY_ENV, raising=False)
    _checkpoint(tmp_path)
    assert verify_release(tmp_path) is False
    (tmp_path / MANIFEST_FILENAME).write_bytes(b"{}")
    (tmp_path / SIGNATURE_FILENAME).write_bytes(b"ab" * 32)
    with pytest.raises(RuntimeError, match=SIGNING_KEY_ENV):
        verify_release(tmp_path)


def test_whitespace_only_key_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(SIGNING_KEY_ENV, "")
    _checkpoint(tmp_path)
    with pytest.raises(RuntimeError, match=SIGNING_KEY_ENV):
        sign_release(tmp_path)


def test_nested_metadata_names_are_hashed_as_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only the two ROOT-level metadata files are exempt — a nested
    ``release.sig`` is ordinary payload and its substitution must fail
    verification."""
    monkeypatch.setenv(SIGNING_KEY_ENV, "synthetic-signing-key")
    _checkpoint(tmp_path)
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / SIGNATURE_FILENAME).write_bytes(b"nested payload")
    sign_release(tmp_path)
    assert verify_release(tmp_path) is True
    (nested / SIGNATURE_FILENAME).write_bytes(b"swapped nested payload")
    assert verify_release(tmp_path) is False


def test_transplanted_attestation_still_checks_contents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A manifest+sig transplanted wholesale into another directory names
    the same artifact paths — the digests must still match real bytes."""
    monkeypatch.setenv(SIGNING_KEY_ENV, "synthetic-signing-key")
    source = tmp_path / "src"
    source.mkdir()
    _checkpoint(source)
    sign_release(source)
    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / MANIFEST_FILENAME).write_bytes((source / MANIFEST_FILENAME).read_bytes())
    (clone / SIGNATURE_FILENAME).write_bytes((source / SIGNATURE_FILENAME).read_bytes())
    (clone / "weights.bin").write_bytes(b"SYNTHETIC weights")
    (clone / "config.json").write_bytes(b'{"model": "SYNTHETIC"}')
    assert verify_release(clone) is True
    (clone / "weights.bin").write_bytes(b"tampered weights")
    assert verify_release(clone) is False


def test_metadata_only_dir_is_not_signable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(SIGNING_KEY_ENV, "synthetic-signing-key")
    (tmp_path / MANIFEST_FILENAME).write_bytes(b"{}")
    with pytest.raises(ValueError, match="no artifacts"):
        build_manifest(tmp_path)
    with pytest.raises(ValueError, match="no artifacts"):
        sign_release(tmp_path)


def test_artifact_added_after_signing_fails_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(SIGNING_KEY_ENV, "synthetic-signing-key")
    _checkpoint(tmp_path)
    sign_release(tmp_path)
    (tmp_path / "stowaway.bin").write_bytes(b"unlisted bytes")
    assert verify_release(tmp_path) is False


def test_manifest_signed_for_other_bytes_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Replacing the manifest bytes after signing invalidates the HMAC —
    the sig authenticates the manifest, not just its presence."""
    monkeypatch.setenv(SIGNING_KEY_ENV, "synthetic-signing-key")
    _checkpoint(tmp_path)
    sign_release(tmp_path)
    (tmp_path / MANIFEST_FILENAME).write_bytes(b'{"checkpoint_dir":"x","artifacts":{"a":"b"}}')
    assert verify_release(tmp_path) is False


@pytest.mark.parametrize("metadata_name", [SIGNATURE_FILENAME, MANIFEST_FILENAME])
def test_signing_never_clobbers_a_symlinked_metadata_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, metadata_name: str
) -> None:
    """A pre-planted ``release.sig``/``release.manifest.json`` symlink must
    refuse, not resolve and clobber the victim file outside the checkpoint."""
    monkeypatch.setenv(SIGNING_KEY_ENV, "synthetic-signing-key")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _checkpoint(ckpt)
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"outside-bytes")
    (ckpt / metadata_name).symlink_to(victim)
    with pytest.raises(ValueError, match="regular files|symlink"):
        sign_release(ckpt)
    assert victim.read_bytes() == b"outside-bytes"
