"""Tests for key_rotation — dual-signed gate-key lineage over the spine."""

from __future__ import annotations

import json
from pathlib import Path

from quant_fund.research.checkpoint_chain import checkpoint_spine
from quant_fund.research.gate_signatures import generate_keypair, sign_pins
from quant_fund.research.integrity_checkpoint import write_checkpoint
from quant_fund.research.key_rotation import (
    load_keyring,
    rotate_key,
    rotation_contract_errors,
    verify_rotations,
)


def _repo(tmp_path: Path) -> tuple[Path, str, str]:
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
    sign_pins(tmp_path, priv, pub)
    return tmp_path, priv, pub


def test_no_rotations_neutral(tmp_path: Path) -> None:
    root, _priv, _pub = _repo(tmp_path)
    res = verify_rotations(root)
    assert res["ok"] is True
    assert res["verdict"] == "no_rotations"
    assert res["n_rotations"] == 0


def test_honest_rotation_verifies(tmp_path: Path) -> None:
    root, priv, pub = _repo(tmp_path)
    write_checkpoint(root, [(priv, pub)])
    _new_priv, new_pub = generate_keypair()
    out = rotate_key(root, priv, _new_priv, reason="test")
    assert out.name.startswith("rotation_")
    res = verify_rotations(root)
    # terminus is the new key but live pub not yet swapped
    assert res["verdict"] == "broken"
    assert "live_key_not_terminus" in res["errors"]
    # install the new key (what an operator does next)
    (root / "quality/gate_signing.pub").write_text(new_pub + "\n")
    res = verify_rotations(root)
    assert res["ok"] is True, res["errors"]
    assert res["verdict"] == "intact"
    assert res["n_rotations"] == 1
    assert res["lineage"][0]["reason"] == "test"


def test_spine_survives_rotation_via_keyring(tmp_path: Path) -> None:
    """Old checkpoints stay valid after the live key rotates — the spine
    verifies each record under its era's authorized key."""
    root, priv, _pub = _repo(tmp_path)
    for _ in range(3):
        write_checkpoint(root, [(priv, _pub)])
    new_priv, new_pub = generate_keypair()
    rotate_key(root, priv, new_priv)
    (root / "quality/gate_signing.pub").write_text(new_pub + "\n")
    # fresh checkpoints under the new key extend the same spine
    write_checkpoint(root, [(new_priv, new_pub)])
    res = checkpoint_spine(root)
    assert res["ok"] is True, res["errors"]
    assert res["spine_length"] == 4
    ring = load_keyring(root)
    assert len(ring) == 2


def test_forged_rotation_fails(tmp_path: Path) -> None:
    """A rotation record where the 'old' side isn't the real old key."""
    root, _priv, _pub = _repo(tmp_path)
    write_checkpoint(root, [(_priv, _pub)])
    attacker_priv, _attacker_pub = generate_keypair()
    _new_priv, _new_pub = generate_keypair()
    # craft the record manually — rotate_key refuses wrong old key, so forge
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from quant_fund.utils.hashing import canonical_json_bytes

    old_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(attacker_priv))
    new_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(_new_priv))
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    old_pub_forge = old_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    new_pub = new_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    from quant_fund.research.gate_signatures import key_id

    payload = {
        "at": "2026-01-01T00:00:00+00:00",
        "reason": "forged",
        "old_key_id": key_id(old_pub_forge),
        "new_key_id": key_id(new_pub),
        "old_pubkey": old_pub_forge,
        "new_pubkey": new_pub,
    }
    body = {
        "schema": "key_rotation.v1",
        "payload": payload,
        "old_signature": old_key.sign(canonical_json_bytes(payload)).hex(),
        "new_signature": new_key.sign(canonical_json_bytes(payload)).hex(),
    }
    (root / "quality/rotation_forge.json").write_text(json.dumps(body) + "\n")
    res = verify_rotations(root)
    assert res["ok"] is False
    assert "unanchored_genesis" in res["errors"]


def test_wrong_old_key_refused(tmp_path: Path) -> None:
    root, priv, _pub = _repo(tmp_path)
    attacker_priv, _ = generate_keypair()
    try:
        rotate_key(root, attacker_priv, priv)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("forged rotation accepted")


def test_malformed_rotation_fails(tmp_path: Path) -> None:
    root, _priv, _pub = _repo(tmp_path)
    (root / "quality/rotation_bad.json").write_text("{not json")
    res = verify_rotations(root)
    assert res["ok"] is False
    assert any("rotation_malformed" in e for e in res["errors"])


def test_contract_coherence(tmp_path: Path) -> None:
    root, priv, pub = _repo(tmp_path)
    write_checkpoint(root, [(priv, pub)])
    new_priv, new_pub = generate_keypair()
    rotate_key(root, priv, new_priv)
    (root / "quality/gate_signing.pub").write_text(new_pub + "\n")
    res = verify_rotations(root)
    assert rotation_contract_errors(res) == []
    bad = dict(res)
    bad["ok"] = True
    bad["errors"] = ["ghost"]
    assert "ok_with_errors" in rotation_contract_errors(bad)
