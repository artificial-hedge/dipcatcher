"""Tests for integrity_checkpoint — the portable signed pin-state snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from quant_fund.research.gate_signatures import generate_keypair, sign_pins
from quant_fund.research.integrity_checkpoint import (
    PINNED_FILES,
    checkpoint_contract_errors,
    checkpoint_state,
    verify_checkpoint,
    write_checkpoint,
)


def _repo(tmp_path: Path) -> tuple[Path, str, str]:
    """A minimal repo with real pins + pubkey; returns (root, priv, pub)."""
    q = tmp_path / "quality"
    q.mkdir(parents=True)
    (q / "crown_jewels.json").write_text('{"pin": "crown"}\n')
    (q / "epoch_heads.json").write_text(
        json.dumps(
            {
                "schema": "epoch_heads.v1",
                "heads": {
                    "receipts/*.json": {
                        "receipt": "corpus_epoch_abc123.json",
                        "sha256": "0" * 64,
                    }
                },
            }
        )
        + "\n"
    )
    priv, pub = generate_keypair()
    sign_pins(tmp_path, priv, pub)  # writes gate_pins.sig + gate_signing.pub
    return tmp_path, priv, pub


def test_write_verify_round_trip(tmp_path: Path) -> None:
    root, priv, pub = _repo(tmp_path)
    write_checkpoint(root, priv, pub)
    res = verify_checkpoint(root)
    assert res["ok"] is True
    assert res["signed"] is True
    assert res["current"] is True


def test_missing_checkpoint_is_neutral(tmp_path: Path) -> None:
    root, _priv, _pub = _repo(tmp_path)
    res = verify_checkpoint(root)
    assert res == {"ok": True, "signed": False, "current": False, "errors": []}


def test_stale_checkpoint_stays_authentic(tmp_path: Path) -> None:
    root, priv, pub = _repo(tmp_path)
    write_checkpoint(root, priv, pub)
    # Pin state moves on (a stamp-epochs run): checkpoint authentic, not current.
    (root / "quality/crown_jewels.json").write_text('{"new": "state"}\n')
    res = verify_checkpoint(root)
    assert res["ok"] is True
    assert res["current"] is False


def test_forged_checkpoint_fails(tmp_path: Path) -> None:
    root, priv, pub = _repo(tmp_path)
    write_checkpoint(root, priv, pub)
    body = json.loads((root / "quality/checkpoint.json").read_text())
    # Attacker rewrites the recorded heads — cannot re-sign with our key.
    body["payload"]["heads"]["receipts/*.json"]["receipt"] = "corpus_epoch_evil.json"
    (root / "quality/checkpoint.json").write_text(json.dumps(body, indent=2, sort_keys=True))
    res = verify_checkpoint(root)
    assert res["ok"] is False
    assert "signature_invalid" in res["errors"]


def test_malformed_checkpoint_fails(tmp_path: Path) -> None:
    root, _priv, _pub = _repo(tmp_path)
    (root / "quality/checkpoint.json").write_text("{not json")
    res = verify_checkpoint(root)
    assert res["ok"] is False
    assert res["errors"] == ["checkpoint_malformed"]


def test_state_records_heads_and_pins(tmp_path: Path) -> None:
    root, _priv, _pub = _repo(tmp_path)
    state = checkpoint_state(root)
    assert state["schema"] == "integrity_checkpoint.v1"
    assert set(state["pins"]) == set(PINNED_FILES)
    assert state["heads"]["receipts/*.json"]["receipt"] == "corpus_epoch_abc123.json"
    assert state["missing_pins"] == []


def test_missing_pin_file_recorded(tmp_path: Path) -> None:
    root, _priv, _pub = _repo(tmp_path)
    (root / "gate_pins.sig").unlink()
    state = checkpoint_state(root)
    assert state["missing_pins"] == ["gate_pins.sig"]


def test_contract_clean(tmp_path: Path) -> None:
    root, _priv, _pub = _repo(tmp_path)
    assert checkpoint_contract_errors(checkpoint_state(root)) == []


def test_contract_catches_forgery() -> None:
    assert checkpoint_contract_errors({"schema": "other"}) == ["schema_mismatch"]
    assert "pins_missing" in checkpoint_contract_errors({"schema": "integrity_checkpoint.v1"})
    bad = {
        "schema": "integrity_checkpoint.v1",
        "pins": {rel: "zz" for rel in PINNED_FILES},
        "heads": {"receipts/*.json": {"receipt": "x.json", "sha256": "0" * 64}},
    }
    errs = checkpoint_contract_errors(bad)
    assert all(e.startswith("pin_digest_malformed:") for e in errs)


def test_verify_repo_carries_checkpoint_gate(tmp_path: Path) -> None:
    # The capstone gate must exist and stay neutral on an unsigned tree.
    from quant_fund.research.repo_integrity import verify_repo

    res = verify_repo(tmp_path)
    assert "checkpoint" in res["gates"]
    assert res["gates"]["checkpoint"]["signed"] is False


def test_checkpoint_chains_to_predecessor(tmp_path: Path) -> None:
    root, priv, pub = _repo(tmp_path)
    write_checkpoint(root, priv, pub)
    first = (root / "quality/checkpoint.json").read_bytes()
    write_checkpoint(root, priv, pub)
    second = json.loads((root / "quality/checkpoint.json").read_text())
    assert second["payload"]["prev_sha256"] == hashlib.sha256(first).hexdigest()
    assert verify_checkpoint(root)["ok"] is True


def test_prev_must_be_witnessed_when_proofs_exist(tmp_path: Path) -> None:
    root, priv, pub = _repo(tmp_path)
    write_checkpoint(root, priv, pub)
    first_sha = hashlib.sha256((root / "quality/checkpoint.json").read_bytes()).hexdigest()
    # A committed witness proof for the first checkpoint.
    w = root / "quality/witness"
    w.mkdir(parents=True)
    (w / "checkpoint.json_1.json").write_text(json.dumps({"target": {"sha256": first_sha}}))
    write_checkpoint(root, priv, pub)
    assert verify_checkpoint(root)["ok"] is True
    # An attacker rewrites prev to an unwitnessed digest — even re-signing
    # with the real key can't make it one of the public witnesses.
    body = json.loads((root / "quality/checkpoint.json").read_text())
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    body["payload"]["prev_sha256"] = "f" * 64
    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv))
    from quant_fund.utils.hashing import canonical_json_bytes

    body["signature"] = key.sign(canonical_json_bytes(body["payload"])).hex()
    (root / "quality/checkpoint.json").write_text(json.dumps(body, indent=2, sort_keys=True))
    res = verify_checkpoint(root)
    assert res["ok"] is False
    assert "prev_not_witnessed" in res["errors"]


def test_contract_accepts_genesis_and_valid_prev(tmp_path: Path) -> None:
    root, _priv, _pub = _repo(tmp_path)
    state = checkpoint_state(root)
    assert state["prev_sha256"] is None  # genesis
    assert checkpoint_contract_errors(state) == []
    state["prev_sha256"] = "x" * 64
    assert checkpoint_contract_errors(state) == []
    state["prev_sha256"] = "tooshort"
    assert "prev_sha256_malformed" in checkpoint_contract_errors(state)


def test_superseded_checkpoint_archived_and_walkable(tmp_path: Path) -> None:
    root, priv, pub = _repo(tmp_path)
    write_checkpoint(root, priv, pub)
    first_bytes = (root / "quality/checkpoint.json").read_bytes()
    write_checkpoint(root, priv, pub)
    archive = root / "quality/checkpoints"
    archived = list(archive.glob("*.json"))
    assert len(archived) == 1
    assert archived[0].read_bytes() == first_bytes
    # The chain walks: current -> archived predecessor.
    second = json.loads((root / "quality/checkpoint.json").read_text())
    prev = second["payload"]["prev_sha256"]
    assert hashlib.sha256(archived[0].read_bytes()).hexdigest() == prev
    assert verify_checkpoint(root)["ok"] is True


def test_unarchived_prev_fails_when_archive_exists(tmp_path: Path) -> None:
    root, priv, pub = _repo(tmp_path)
    write_checkpoint(root, priv, pub)
    write_checkpoint(root, priv, pub)  # creates the archive dir
    # Rotate again but delete the archive of the just-superseded checkpoint:
    # the declared prev was never retained -> fail.
    from quant_fund.utils.hashing import hash_bytes as _hb

    cur_prev = hashlib.sha256((root / "quality/checkpoint.json").read_bytes()).hexdigest()
    write_checkpoint(root, priv, pub)
    for f in (root / "quality/checkpoints").glob("*.json"):
        if _hb(f.read_bytes()) == cur_prev:
            f.unlink()
    res = verify_checkpoint(root)
    assert "prev_not_archived" in res["errors"]
