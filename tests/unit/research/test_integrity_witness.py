"""Tests for the Rekor transparency-log witness lane.

The committed proof ``quality/witness/checkpoint.json_<index>.json`` is a
real Rekor entry — the happy-path tests exercise the full offline
verification (RFC 6962 inclusion, Rekor SET + checkpoint-note signatures,
witness attribution sig) against live-recorded cryptographic material, so
no network and no fixtures-with-fake-keys are needed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from quant_fund.research.integrity_witness import (
    REKOR_PUBKEY_PATH,
    WITNESS_DIR,
    WITNESS_PUBKEY_PATH,
    verify_witness_file,
    verify_witnesses,
    witness_contract_errors,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _committed_proofs() -> list[Path]:
    return sorted((REPO_ROOT / WITNESS_DIR).glob("checkpoint.json_*.json"))


def _proof_path() -> Path:
    return _committed_proofs()[-1]


PROOF_NAME = _proof_path().name if _committed_proofs() else ""

requires_proof = pytest.mark.skipif(
    not _committed_proofs(), reason="committed witness proof not present on this branch"
)


@pytest.fixture
def proof_repo(tmp_path: Path) -> Path:
    """A tmp root holding the real committed proof + its target + both pubkeys."""
    proof_src = _proof_path()
    root = tmp_path / "repo"
    (root / WITNESS_DIR).mkdir(parents=True)
    (root / "quality").mkdir(exist_ok=True)
    shutil.copy(proof_src, root / WITNESS_DIR / PROOF_NAME)
    shutil.copy(REPO_ROOT / "quality/checkpoint.json", root / "quality/checkpoint.json")
    shutil.copy(REPO_ROOT / WITNESS_PUBKEY_PATH, root / WITNESS_PUBKEY_PATH)
    shutil.copy(REPO_ROOT / REKOR_PUBKEY_PATH, root / REKOR_PUBKEY_PATH)
    return root


def _mutate_proof(root: Path, fn) -> None:
    p = root / WITNESS_DIR / PROOF_NAME
    record = json.loads(p.read_text())
    fn(record)
    p.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


@requires_proof
def test_committed_proof_verifies_on_the_real_tree() -> None:
    """The committed Rekor proof verifies against the live repo checkout."""
    res = verify_witness_file(REPO_ROOT, Path(WITNESS_DIR) / PROOF_NAME)
    assert res["errors"] == []
    assert res["ok"] is True


def test_verify_witnesses_neutral_without_dir(tmp_path: Path) -> None:
    res = verify_witnesses(tmp_path)
    assert res == {"ok": True, "witnessed": False, "errors": [], "proofs": {}}


@requires_proof
def test_proof_round_trip_in_tmp_root(proof_repo: Path) -> None:
    res = verify_witnesses(proof_repo)
    assert res["witnessed"] is True
    assert res["errors"] == []
    assert res["proofs"][PROOF_NAME]["current"] is True


@requires_proof
def test_digest_mismatch_flags_edited_target_field(proof_repo: Path) -> None:
    def edit(rec):
        rec["target"]["sha256"] = "0" * 64

    _mutate_proof(proof_repo, edit)
    res = verify_witnesses(proof_repo)
    assert any("digest_mismatch" in e for e in res["errors"])


@requires_proof
def test_body_tamper_breaks_leaf_binding(proof_repo: Path) -> None:
    def edit(rec):
        rec["rekor"]["body_b64"] = "e30="  # "{}"

    _mutate_proof(proof_repo, edit)
    res = verify_witnesses(proof_repo)
    assert any("leaf_uuid_mismatch" in e or "digest_mismatch" in e for e in res["errors"])


@requires_proof
def test_forged_set_signature_fails(proof_repo: Path) -> None:
    def edit(rec):
        rec["rekor"]["signed_entry_timestamp"] = "AAAA"

    _mutate_proof(proof_repo, edit)
    res = verify_witnesses(proof_repo)
    assert any("set_signature_invalid" in e for e in res["errors"])


@requires_proof
def test_forged_inclusion_root_fails(proof_repo: Path) -> None:
    def edit(rec):
        rec["rekor"]["inclusion_proof"]["root_hash"] = "0" * 64

    _mutate_proof(proof_repo, edit)
    res = verify_witnesses(proof_repo)
    assert any("inclusion_root_mismatch" in e for e in res["errors"])


@requires_proof
def test_missing_rekor_pubkey_fails_closed(proof_repo: Path) -> None:
    (proof_repo / REKOR_PUBKEY_PATH).unlink()
    res = verify_witnesses(proof_repo)
    assert any("rekor_pubkey_missing" in e for e in res["errors"])


@requires_proof
def test_stale_target_is_flagged_not_rejected(proof_repo: Path) -> None:
    """Mutating the checkpoint after witnessing → current=False, ok=True."""
    (proof_repo / "quality/checkpoint.json").write_text("{}")
    res = verify_witnesses(proof_repo)
    assert res["errors"] == []
    assert res["proofs"][PROOF_NAME]["current"] is False


def test_malformed_proof_file_fails(tmp_path: Path) -> None:
    (tmp_path / WITNESS_DIR).mkdir(parents=True)
    (tmp_path / WITNESS_DIR / "junk.json").write_text("{")
    res = verify_witnesses(tmp_path)
    assert any("proof_malformed" in e for e in res["errors"])


@requires_proof
def test_witness_contract_accepts_committed_proof() -> None:
    payload = json.loads(_proof_path().read_text())
    assert witness_contract_errors(payload) == []


def test_witness_contract_rejects_forged_shape() -> None:
    assert "rekor_missing" in witness_contract_errors(
        {"schema": "integrity_witness.v1", "target": {"sha256": "x" * 64}}
    )
    assert "schema_mismatch" in witness_contract_errors({"schema": "other"})


def test_consistency_kat_on_real_rekor_proof() -> None:
    """Offline KAT: a real RFC 6962 consistency proof fetched from Rekor must
    verify between the two committed proofs' recorded tree sizes; a flipped
    root bit must fail."""
    from quant_fund.research.integrity_witness import _consistency_ok

    kat = json.loads(
        (REPO_ROOT / "tests/unit/research/fixtures/rekor_consistency_kat.json").read_text()
    )
    path = [bytes.fromhex(h) for h in kat["hashes"]]
    assert _consistency_ok(
        kat["first_size"],
        bytes.fromhex(kat["first_root"]),
        kat["second_size"],
        bytes.fromhex(kat["second_root"]),
        path,
    )
    bad_root = bytes.fromhex("00" + kat["second_root"][2:])
    assert not _consistency_ok(
        kat["first_size"], bytes.fromhex(kat["first_root"]), kat["second_size"], bad_root, path
    )
    # Truncated path must not verify either.
    assert not _consistency_ok(
        kat["first_size"],
        bytes.fromhex(kat["first_root"]),
        kat["second_size"],
        bytes.fromhex(kat["second_root"]),
        path[:-1],
    )


@pytest.mark.network
def test_online_consistency_against_live_rekor() -> None:
    """Live: every committed proof's tree must still be a prefix of Rekor's
    current signed tree head (consistency path + STH signature)."""
    from quant_fund.research.integrity_witness import verify_witness_online

    res = verify_witness_online(REPO_ROOT)
    assert res["online"] is True
    assert res["errors"] == []
    assert all(res["consistent"].values())
