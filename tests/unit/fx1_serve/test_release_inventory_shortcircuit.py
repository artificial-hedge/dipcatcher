"""SYNTHETIC checks: reject invalid inventories before reading model weights."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pytest

from fx1.serve import signing

_TEST_KEY = "SYNTHETIC-inventory-test-key"


@pytest.fixture
def checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(signing.SIGNING_KEY_ENV, _TEST_KEY)
    (tmp_path / "weights.bin").write_bytes(b"SYNTHETIC weights")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    signing.sign_release(tmp_path)
    return tmp_path


@pytest.mark.parametrize("mutation", ["add", "delete", "rename", "directory"])
def test_inventory_mismatch_does_not_hash_artifacts(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    weights = checkpoint / "weights.bin"
    if mutation == "add":
        (checkpoint / "unlisted.bin").write_bytes(b"SYNTHETIC extra")
    elif mutation == "rename":
        weights.rename(checkpoint / "renamed.bin")
    else:
        weights.unlink()
        if mutation == "directory":
            weights.mkdir()

    def unexpected_hash(path: Path) -> str:
        pytest.fail(f"invalid inventory triggered artifact hashing: {path}")

    monkeypatch.setattr(signing, "_hash_file", unexpected_hash)
    assert signing.verify_release(checkpoint) is False


@pytest.mark.parametrize("name", ["../outside.bin", "/outside.bin"])
def test_nonlocal_manifest_path_is_rejected_before_hashing(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    raw = json.dumps(
        {"checkpoint_dir": str(checkpoint), "artifacts": {name: "a" * 64}}
    ).encode()
    (checkpoint / signing.MANIFEST_FILENAME).write_bytes(raw)
    (checkpoint / signing.SIGNATURE_FILENAME).write_text(
        hmac.new(_TEST_KEY.encode(), raw, hashlib.sha256).hexdigest(), encoding="ascii"
    )

    def unexpected_hash(path: Path) -> str:
        pytest.fail(f"nonlocal manifest triggered artifact hashing: {path}")

    monkeypatch.setattr(signing, "_hash_file", unexpected_hash)
    assert signing.verify_release(checkpoint) is False


def test_matching_inventory_still_hashes_each_artifact_on_every_verification(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = signing._hash_file
    hashed: list[Path] = []

    def tracked_hash(path: Path) -> str:
        hashed.append(path)
        return original(path)

    monkeypatch.setattr(signing, "_hash_file", tracked_hash)
    expected = {checkpoint / "weights.bin", checkpoint / "config.json"}
    for _ in range(2):
        hashed.clear()
        assert signing.verify_release(checkpoint) is True
        assert set(hashed) == expected
        assert len(hashed) == len(expected)
    (checkpoint / "weights.bin").write_bytes(b"tampered")
    assert signing.verify_release(checkpoint) is False
