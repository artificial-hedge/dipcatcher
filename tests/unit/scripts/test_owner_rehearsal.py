"""The owner rehearsal must stay outside the real checkout."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from scripts.owner_rehearsal import (
    RehearsalRefusal,
    assert_throwaway,
    historical_receipt_bytes,
    public_hex,
    receipt_drift,
    require_copy,
    restore_stamped_receipts,
    retarget_data_manifest,
    unlink_under,
    write_quorum_registry,
)

from quant_fund.research.gate_signatures import generate_keypair, quorum_registry_errors
from quant_fund.utils.hashing import hash_bytes

REPO = Path(__file__).resolve().parents[3]


def test_throwaway_refuses_the_source_and_its_children(tmp_path: Path) -> None:
    source = tmp_path / "real"
    source.mkdir()
    with pytest.raises(RehearsalRefusal):
        assert_throwaway(source, source)
    nested = source / "nested"
    nested.mkdir()
    with pytest.raises(RehearsalRefusal):
        assert_throwaway(source, nested)
    with pytest.raises(RehearsalRefusal):
        assert_throwaway(nested, source)


def test_throwaway_accepts_a_sibling_directory(tmp_path: Path) -> None:
    source = tmp_path / "real"
    dest = tmp_path / "copy"
    source.mkdir()
    dest.mkdir()
    assert_throwaway(source, dest)


def test_require_copy_needs_the_marker(tmp_path: Path) -> None:
    source = tmp_path / "real"
    dest = tmp_path / "copy"
    source.mkdir()
    dest.mkdir()
    with pytest.raises(RehearsalRefusal):
        require_copy(source, dest)
    (dest / ".owner-rehearsal").write_text("throwaway\n", encoding="utf-8")
    require_copy(source, dest)


def test_unlink_under_rejects_paths_outside_the_copy(tmp_path: Path) -> None:
    copy_root = tmp_path / "copy" / "quality"
    outside = tmp_path / "real" / "quality"
    copy_root.mkdir(parents=True)
    outside.mkdir(parents=True)
    victim = outside / "checkpoint.json"
    victim.write_text("keep\n", encoding="utf-8")
    with pytest.raises(RehearsalRefusal):
        unlink_under(copy_root, victim)
    assert victim.read_text(encoding="utf-8") == "keep\n"
    link = copy_root / "checkpoint.json"
    link.symlink_to(victim)
    with pytest.raises(RehearsalRefusal):
        unlink_under(copy_root, link)
    assert victim.is_file()


def test_public_hex_matches_the_seed(tmp_path: Path) -> None:
    seed, pub = generate_keypair()
    assert public_hex(seed) == pub
    registry = tmp_path / "quorum.json"
    write_quorum_registry(pub, registry)
    body = json.loads(registry.read_text(encoding="utf-8"))
    assert body["threshold"] == 1
    assert body["keys"][0]["pubkey"] == pub
    assert quorum_registry_errors(body) == []


def test_retarget_points_manifest_paths_at_the_copy(tmp_path: Path) -> None:
    source = tmp_path / "real"
    dest = tmp_path / "copy"
    source.mkdir()
    dest.mkdir()
    (dest / ".owner-rehearsal").write_text("throwaway\n", encoding="utf-8")
    manifest = dest / "data" / "metadata" / "data_manifest.json"
    manifest.parent.mkdir(parents=True)
    original = {
        "schema_version": 1,
        "source": "synthetic",
        "artifacts": {"bars": {"path": str(source / "data/raw/bars.parquet"), "rows": 1}},
    }
    manifest.write_text(json.dumps(original), encoding="utf-8")
    assert retarget_data_manifest(source, dest) is True
    body = json.loads(manifest.read_text(encoding="utf-8"))
    assert body["artifacts"]["bars"]["path"] == str((dest / "data/raw/bars.parquet").resolve())
    assert "receipt_sha256" in body
    assert retarget_data_manifest(source, dest) is False


def test_owner_steps_sign_after_the_new_public_key_and_witness_the_one_checkpoint() -> None:
    text = (REPO / "OWNER_STEPS.md").read_text(encoding="utf-8")
    order = [
        "make rotate-key",
        "printf '%s\\n' \"$new_pub\" > quality/gate_signing.pub",
        "quorum-rotate",
        "crown-jewels --write",
        "--restore-stamped-receipts",
        "make stamp-epochs",
        "make sign-pins",
        "make anchor-pins",
        "make checkpoint\n",
        "make anchor-checkpoint",
        "make witness-checkpoint",
        "git tag -a v0.4.0",
        "owner rehearsal: local tag",
    ]
    positions = [text.index(token) for token in order]
    assert positions == sorted(positions)
    assert "git push origin v0.4.0" in text


def test_sign_pins_uses_the_quorum_and_anchor_does_not_resign() -> None:
    text = (REPO / "Makefile").read_text(encoding="utf-8")
    sign = text.split("sign-pins:", 1)[1].split("\n\n", 1)[0]
    anchor = text.split("anchor-checkpoint:", 1)[1].split("\n\n", 1)[0]
    assert "quorum-sign" in sign
    assert "anchor-timestamp --file quality/checkpoint.json" in anchor
    assert "checkpoint --anchor" not in anchor
    assert "owner-rehearsal:" in text


def test_only_the_retrieval_audit_drifts_and_git_has_the_stamp() -> None:
    drifted = receipt_drift(REPO)
    assert [name for name, _digest in drifted] == ["fx1_retrieval_audit.json"]
    name, digest = drifted[0]
    blob = historical_receipt_bytes(REPO, name, digest)
    assert blob is not None
    assert hash_bytes(blob) == digest
    live = REPO / "receipts" / name
    assert hash_bytes(live.read_bytes()) != digest


def test_restore_puts_back_the_stamped_digest(tmp_path: Path) -> None:
    repo = _receipt_repo(tmp_path)
    target = repo / "receipts" / "fx1_retrieval_audit.json"
    original = target.read_bytes()
    digest = hash_bytes(original)
    _commit_all(repo, "stamp")
    target.write_bytes(b'{"schema":"audit","n":2}\n')
    _commit_all(repo, "drift")
    _write_heads(repo, "fx1_retrieval_audit.json", digest)
    heads = (repo / "quality" / "epoch_heads.json").read_bytes()
    restored, missing = restore_stamped_receipts(repo)
    assert restored == ["fx1_retrieval_audit.json"]
    assert missing == []
    assert target.read_bytes() == original
    assert hash_bytes(target.read_bytes()) == digest
    assert (repo / "quality" / "epoch_heads.json").read_bytes() == heads
    assert restore_stamped_receipts(repo) == ([], [])


def test_allowlisted_absence_is_not_drift(tmp_path: Path) -> None:
    repo = _receipt_repo(tmp_path)
    target = repo / "receipts" / "fx1_retrieval_audit.json"
    digest = hash_bytes(target.read_bytes())
    target.unlink()
    _write_heads(repo, "fx1_retrieval_audit.json", digest)
    allow = repo / "quality" / "epoch_allowed_removals.json"
    allow.write_text(json.dumps({"fx1_retrieval_audit.json": digest}), encoding="utf-8")
    assert receipt_drift(repo) == []
    assert restore_stamped_receipts(repo) == ([], [])
    assert not target.exists()


def test_restore_leaves_a_member_with_no_historical_blob(tmp_path: Path) -> None:
    repo = _receipt_repo(tmp_path)
    target = repo / "receipts" / "fx1_retrieval_audit.json"
    drifted = target.read_bytes()
    _commit_all(repo, "drift-only")
    _write_heads(repo, "fx1_retrieval_audit.json", "ab" * 32)
    restored, missing = restore_stamped_receipts(repo)
    assert restored == []
    assert missing == ["fx1_retrieval_audit.json"]
    assert target.read_bytes() == drifted


def test_restore_refuses_a_symlink_and_a_path_escape(tmp_path: Path) -> None:
    repo = _receipt_repo(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"keep\n")
    link = repo / "receipts" / "fx1_retrieval_audit.json"
    link.unlink()
    link.symlink_to(outside)
    _write_heads(repo, "fx1_retrieval_audit.json", "cd" * 32)
    with pytest.raises(RehearsalRefusal):
        restore_stamped_receipts(repo)
    assert outside.read_bytes() == b"keep\n"
    _write_heads(repo, "../outside.json", "cd" * 32)
    with pytest.raises(RehearsalRefusal):
        restore_stamped_receipts(repo)
    assert outside.read_bytes() == b"keep\n"


def _receipt_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "receipts").mkdir(parents=True)
    (repo / "quality").mkdir()
    (repo / "receipts" / "fx1_retrieval_audit.json").write_bytes(b'{"schema":"audit","n":1}\n')
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    return repo


def _commit_all(repo: Path, message: str) -> None:
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        }
    )
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo,
        check=True,
        capture_output=True,
        env=env,
    )


def _write_heads(repo: Path, name: str, digest: str) -> None:
    epoch_name = "corpus_epoch_abc.json"
    epoch = {"members": [{"name": name, "sha256": digest}]}
    (repo / "receipts" / epoch_name).write_text(json.dumps(epoch), encoding="utf-8")
    pin = {
        "schema": "epoch_heads.v1",
        "heads": {"receipts/*.json": {"receipt": epoch_name, "sha256": "11" * 32}},
    }
    (repo / "quality" / "epoch_heads.json").write_text(json.dumps(pin), encoding="utf-8")
