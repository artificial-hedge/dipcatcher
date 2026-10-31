"""Tests for quorum_rotation — the M-of-N signer-set lineage.

The registry file authorizes v2 checkpoints, but is itself only bytes on
disk; these tests pin the ceremony that makes a registry swap detectable:
rotation records signed to the *predecessor* quorum, digest-chained, with
the live registry bound to the chain terminus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.research.gate_signatures import (
    generate_keypair,
    init_quorum,
    key_id,
    registry_file_bytes,
    registry_sha256,
    sign_pins,
)
from quant_fund.research.integrity_checkpoint import (
    verify_checkpoint,
    write_checkpoint,
)
from quant_fund.research.quorum_rotation import (
    QUORUM_ROTATION_DIR,
    QUORUM_ROTATION_SCHEMA,
    rotate_quorum,
    verify_quorum_rotations,
)


def _repo(tmp_path: Path) -> Path:
    """Minimal pinned repo (same fixture shape as test_integrity_checkpoint)."""
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
    return tmp_path


def _registry(pairs: list[tuple[str, str]], threshold: int) -> dict:
    return {
        "schema": "gate_quorum.v1",
        "threshold": threshold,
        "keys": [{"key_id": key_id(pub), "pubkey": pub} for _priv, pub in pairs],
    }


def _repo_with_quorum(
    tmp_path: Path, n_keys: int = 2, threshold: int = 2
) -> tuple[Path, list[tuple[str, str]]]:
    root = _repo(tmp_path)
    pairs = [generate_keypair() for _ in range(n_keys)]
    init_quorum(root, [pub for _priv, pub in pairs], threshold=threshold)
    return root, pairs


# -- ceremony ----------------------------------------------------------------


def test_rotate_round_trip(tmp_path: Path) -> None:
    root, pairs = _repo_with_quorum(tmp_path)
    write_checkpoint(root, pairs)  # v2 head binds the genesis registry
    new_pairs = [generate_keypair() for _ in range(2)]
    new_reg = _registry(new_pairs, 2)
    rec_path = rotate_quorum(root, new_reg, pairs)
    assert rec_path.parent.name == QUORUM_ROTATION_DIR.name
    body = json.loads(rec_path.read_text())
    assert body["schema"] == QUORUM_ROTATION_SCHEMA
    assert len(body["signatures"]) == 2
    # Live registry is byte-exact the digests bound.
    live = json.loads((root / "quality/gate_quorum.json").read_text())
    assert registry_sha256(live) == body["payload"]["registry_sha256"]
    assert (root / "quality/gate_quorum.json").read_bytes() == registry_file_bytes(new_reg)
    res = verify_quorum_rotations(root)
    assert res["ok"] is True
    assert res["n_rotations"] == 1


def test_rotate_below_threshold_refused(tmp_path: Path) -> None:
    root, pairs = _repo_with_quorum(tmp_path)
    before = (root / "quality/gate_quorum.json").read_bytes()
    with pytest.raises(ValueError, match="unattainable"):
        rotate_quorum(root, _registry([generate_keypair()], 1), pairs[:1])
    assert (root / "quality/gate_quorum.json").read_bytes() == before
    assert not (root / QUORUM_ROTATION_DIR).exists()


def test_rotate_unregistered_signer_refused(tmp_path: Path) -> None:
    root, pairs = _repo_with_quorum(tmp_path)
    rogue = generate_keypair()
    with pytest.raises(ValueError, match="not in current quorum registry"):
        rotate_quorum(root, _registry(pairs, 1), [pairs[0], rogue])


def test_rotate_without_registry_refused(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    with pytest.raises(ValueError, match="genesis uses quorum-init"):
        rotate_quorum(root, _registry([generate_keypair()], 1), [])


def test_rotate_no_records_is_neutral(tmp_path: Path) -> None:
    root, _pairs = _repo_with_quorum(tmp_path)
    res = verify_quorum_rotations(root)
    assert res["ok"] is True
    assert res["verdict"] == "no_rotations"


# -- adversarial -------------------------------------------------------------


def test_live_swap_without_record_flagged(tmp_path: Path) -> None:
    """The attack this lane exists for: substitute the registry outright."""
    root, pairs = _repo_with_quorum(tmp_path)
    write_checkpoint(root, pairs)
    rotate_quorum(root, _registry([generate_keypair()], 1), pairs)
    attacker = _registry([generate_keypair()], 1)
    (root / "quality/gate_quorum.json").write_bytes(registry_file_bytes(attacker))
    res = verify_quorum_rotations(root)
    assert res["ok"] is False
    assert "registry_not_terminus" in res["errors"]


def test_forged_rotation_record_fails(tmp_path: Path) -> None:
    root, pairs = _repo_with_quorum(tmp_path)
    write_checkpoint(root, pairs)
    rotate_quorum(root, _registry([generate_keypair()], 1), pairs)
    # Attacker writes a second record off the same prev, signed by a rogue key.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    from quant_fund.utils.hashing import canonical_json_bytes

    prev_reg = json.loads(next(iter((root / QUORUM_ROTATION_DIR).glob("*.json"))).read_text())[
        "payload"
    ]["prev_registry"]
    rogue_priv, rogue_pub = generate_keypair()
    fake_new = _registry([generate_keypair()], 1)
    payload = {
        "at": "2026-01-01T00:00:00+00:00",
        "reason": "forged",
        "prev_registry": prev_reg,
        "prev_registry_sha256": registry_sha256(prev_reg),
        "registry": fake_new,
        "registry_sha256": registry_sha256(fake_new),
    }
    body = {
        "schema": QUORUM_ROTATION_SCHEMA,
        "algorithm": "ed25519",
        "payload": payload,
        "signatures": [
            {
                "key_id": key_id(rogue_pub),
                "signature": Ed25519PrivateKey.from_private_bytes(bytes.fromhex(rogue_priv))
                .sign(canonical_json_bytes(payload))
                .hex(),
            }
        ],
    }
    (root / QUORUM_ROTATION_DIR / "rotation_forged.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n"
    )
    res = verify_quorum_rotations(root)
    assert res["ok"] is False
    assert any("signer_unknown" in e or "below_quorum" in e for e in res["errors"])
    assert any(e.startswith("rotation_fork") for e in res["errors"])


def test_tampered_record_signature_invalid(tmp_path: Path) -> None:
    root, pairs = _repo_with_quorum(tmp_path)
    write_checkpoint(root, pairs)
    rec_path = rotate_quorum(root, _registry([generate_keypair()], 1), pairs)
    body = json.loads(rec_path.read_text())
    body["payload"]["reason"] = "attacker rewrite"
    rec_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    res = verify_quorum_rotations(root)
    assert res["ok"] is False
    assert any("signature_invalid" in e for e in res["errors"])


def test_unanchored_genesis_flagged(tmp_path: Path) -> None:
    """A rotation whose predecessor never governed anything: no spine record
    binds the claimed prev digest, and no spine signer overlaps its keys."""
    root, pairs = _repo_with_quorum(tmp_path)
    # No write_checkpoint — the prev registry never appears on a spine.
    rotate_quorum(root, _registry([generate_keypair()], 1), pairs)
    res = verify_quorum_rotations(root)
    assert res["ok"] is False
    assert "unanchored_genesis" in res["errors"]


def test_malformed_record_fails_closed(tmp_path: Path) -> None:
    root, pairs = _repo_with_quorum(tmp_path)
    write_checkpoint(root, pairs)
    rotate_quorum(root, _registry([generate_keypair()], 1), pairs)
    (root / QUORUM_ROTATION_DIR / "rotation_garbage.json").write_text("{nope")
    res = verify_quorum_rotations(root)
    assert res["ok"] is False
    assert any("malformed" in e for e in res["errors"])


# -- spine continuity ---------------------------------------------------------


def test_mixed_era_spine_accepts_authorized_rotation(tmp_path: Path) -> None:
    """v2 checkpoints written across a rotation verify under their own era's
    registry, and the record authorizes the digest change between them."""
    from quant_fund.research.checkpoint_chain import checkpoint_spine

    root, pairs = _repo_with_quorum(tmp_path)
    write_checkpoint(root, pairs)  # era A head
    new_pairs = [generate_keypair()]
    rotate_quorum(root, _registry(new_pairs, 1), pairs)  # authorized link
    write_checkpoint(root, new_pairs)  # era B head, archives era A
    res = checkpoint_spine(root)
    assert res["ok"] is True, res["errors"]


def test_spine_flags_unauthorized_registry_change(tmp_path: Path) -> None:
    """A v2 child claiming a different registry with no authorized rotation
    record is the forged-head attack — the spine must refuse it."""
    from quant_fund.research.checkpoint_chain import checkpoint_spine

    root, pairs = _repo_with_quorum(tmp_path)
    write_checkpoint(root, pairs)  # honest era-A head
    # Attacker swaps the registry and forges a head under their own key.
    attacker = generate_keypair()
    (root / "quality/gate_quorum.json").write_bytes(registry_file_bytes(_registry([attacker], 1)))
    write_checkpoint(root, [attacker])  # era-B forged head, archives era A
    res = checkpoint_spine(root)
    assert res["ok"] is False
    assert any(e.startswith("quorum_registry_unauthorized") for e in res["errors"])


def test_head_quorum_field_binds_registry(tmp_path: Path) -> None:
    """verify_checkpoint hard-fails when the claimed registry digest drifts."""
    root, pairs = _repo_with_quorum(tmp_path)
    write_checkpoint(root, pairs)
    body = json.loads((root / "quality/checkpoint.json").read_text())
    claimed = body["payload"]["quorum"]["registry_sha256"]
    assert claimed == registry_sha256(json.loads((root / "quality/gate_quorum.json").read_text()))
    # Claiming a different digest must fail.
    body["payload"]["quorum"]["registry_sha256"] = "0" * 64
    (root / "quality/checkpoint.json").write_text(json.dumps(body))
    res = verify_checkpoint(root)
    assert res["ok"] is False
    assert any("quorum_registry_drift" in e for e in res["errors"])
