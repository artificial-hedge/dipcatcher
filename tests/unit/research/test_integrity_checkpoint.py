"""Tests for integrity_checkpoint — the portable signed pin-state snapshot."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from quant_fund.research.gate_signatures import generate_keypair, sign_pins
from quant_fund.research.integrity_checkpoint import (
    PINNED_FILES,
    checkpoint_contract_errors,
    checkpoint_state,
    verify_checkpoint,
    write_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


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
    write_checkpoint(root, [(priv, pub)])
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
    write_checkpoint(root, [(priv, pub)])
    # Pin state moves on (a stamp-epochs run): checkpoint authentic, not current.
    (root / "quality/crown_jewels.json").write_text('{"new": "state"}\n')
    res = verify_checkpoint(root)
    assert res["ok"] is True
    assert res["current"] is False


def test_forged_checkpoint_fails(tmp_path: Path) -> None:
    root, priv, pub = _repo(tmp_path)
    write_checkpoint(root, [(priv, pub)])
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
    # pins covers every PINNED_FILE present on disk; this fixture has no
    # quorum registry, so it lands in missing_pins instead.
    assert set(state["pins"]) == {rel for rel in PINNED_FILES if (root / rel).exists()}
    assert state["heads"]["receipts/*.json"]["receipt"] == "corpus_epoch_abc123.json"
    assert state["missing_pins"] == ["quality/gate_quorum.json"]


def test_missing_pin_file_recorded(tmp_path: Path) -> None:
    root, _priv, _pub = _repo(tmp_path)
    (root / "gate_pins.sig").unlink()
    state = checkpoint_state(root)
    assert state["missing_pins"] == ["quality/gate_quorum.json", "gate_pins.sig"]


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
    assert sum(e.startswith("pin_digest_malformed:") for e in errs) == len(PINNED_FILES)
    # bad lacks missing_pins — the pins∪missing accounting itself is flagged.
    assert "missing_pins_malformed" in errs


def test_verify_repo_carries_checkpoint_gate(tmp_path: Path) -> None:
    # The capstone gate must exist and stay neutral on an unsigned tree.
    from quant_fund.research.repo_integrity import verify_repo

    res = verify_repo(tmp_path)
    assert "checkpoint" in res["gates"]
    assert res["gates"]["checkpoint"]["signed"] is False


def test_checkpoint_chains_to_predecessor(tmp_path: Path) -> None:
    root, priv, pub = _repo(tmp_path)
    write_checkpoint(root, [(priv, pub)])
    first = (root / "quality/checkpoint.json").read_bytes()
    write_checkpoint(root, [(priv, pub)])
    second = json.loads((root / "quality/checkpoint.json").read_text())
    assert second["payload"]["prev_sha256"] == hashlib.sha256(first).hexdigest()
    assert verify_checkpoint(root)["ok"] is True


def test_prev_must_be_witnessed_when_proofs_exist(tmp_path: Path) -> None:
    root, priv, pub = _repo(tmp_path)
    write_checkpoint(root, [(priv, pub)])
    first_sha = hashlib.sha256((root / "quality/checkpoint.json").read_bytes()).hexdigest()
    # A committed witness proof for the first checkpoint.
    w = root / "quality/witness"
    w.mkdir(parents=True)
    (w / "checkpoint.json_1.json").write_text(json.dumps({"target": {"sha256": first_sha}}))
    write_checkpoint(root, [(priv, pub)])
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
    write_checkpoint(root, [(priv, pub)])
    first_bytes = (root / "quality/checkpoint.json").read_bytes()
    write_checkpoint(root, [(priv, pub)])
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
    write_checkpoint(root, [(priv, pub)])
    write_checkpoint(root, [(priv, pub)])  # creates the archive dir
    # Rotate again but delete the archive of the just-superseded checkpoint:
    # the declared prev was never retained -> fail.
    from quant_fund.utils.hashing import hash_bytes as _hb

    cur_prev = hashlib.sha256((root / "quality/checkpoint.json").read_bytes()).hexdigest()
    write_checkpoint(root, [(priv, pub)])
    for f in (root / "quality/checkpoints").glob("*.json"):
        if _hb(f.read_bytes()) == cur_prev:
            f.unlink()
    res = verify_checkpoint(root)
    assert "prev_not_archived" in res["errors"]


def test_code_attestation_present_and_wellformed(tmp_path: Path) -> None:
    root, _priv, _pub = _repo(tmp_path)
    state = checkpoint_state(root)
    code = state["code"]
    assert set(code) == {"revision", "worktree_sha256"}
    assert checkpoint_contract_errors(state) == []
    # Outside a git checkout both are UNKNOWN and still contract-clean.
    assert code["revision"] in ("UNKNOWN",) or len(code["revision"]) == 40


def test_code_attestation_binds_real_revision() -> None:
    state = checkpoint_state(REPO_ROOT)
    assert (
        state["code"]["revision"]
        == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    )


# -- multisig checkpoints (integrity_checkpoint_sig.v2) ------------------------


def _repo_with_quorum(
    tmp_path: Path, n_keys: int, threshold: int
) -> tuple[Path, list[tuple[str, str]]]:
    """A pinned repo plus a committed gate_quorum.v1 registry.

    Returns (root, [(priv, pub), ...]) — all n keys registered.
    """
    root, _priv, _pub = _repo(tmp_path)
    pairs = [generate_keypair() for _ in range(n_keys)]
    from quant_fund.research.gate_signatures import init_quorum

    init_quorum(root, [pub for _priv, pub in pairs], threshold=threshold)
    return root, pairs


def test_v2_quorum_checkpoint_round_trip(tmp_path: Path) -> None:
    root, pairs = _repo_with_quorum(tmp_path, 2, 2)
    write_checkpoint(root, pairs)
    body = json.loads((root / "quality/checkpoint.json").read_text())
    assert body["schema"] == "integrity_checkpoint_sig.v2"
    assert len(body["signatures"]) == 2
    res = verify_checkpoint(root)
    assert res["ok"] is True
    assert res["current"] is True


def test_v2_below_quorum_refuses_to_write(tmp_path: Path) -> None:
    root, pairs = _repo_with_quorum(tmp_path, 2, 2)
    import pytest

    with pytest.raises(ValueError, match="quorum unattainable"):
        write_checkpoint(root, pairs[:1])
    # A partial checkpoint must never land — the head stays absent.
    assert not (root / "quality/checkpoint.json").exists()


def test_v2_unregistered_signer_refused(tmp_path: Path) -> None:
    root, pairs = _repo_with_quorum(tmp_path, 1, 1)
    rogue = generate_keypair()
    import pytest

    with pytest.raises(ValueError, match="not in quorum registry"):
        write_checkpoint(root, [pairs[0], rogue])


def test_v1_head_under_committed_registry_fails(tmp_path: Path) -> None:
    """Downgrade attack: a single-key envelope cannot satisfy a quorum tree."""
    root, pairs = _repo_with_quorum(tmp_path, 1, 1)
    # Hand-craft a v1 checkpoint signed by the registered key.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from quant_fund.utils.hashing import canonical_json_bytes

    state = checkpoint_state(root)
    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(pairs[0][0]))
    body = {
        "schema": "integrity_checkpoint_sig.v1",
        "algorithm": "ed25519",
        "key_id": pairs[0][1][:16],
        "payload": state,
        "signature": key.sign(canonical_json_bytes(state)).hex(),
    }
    (root / "quality/checkpoint.json").write_text(json.dumps(body))
    res = verify_checkpoint(root)
    assert res["ok"] is False
    assert "checkpoint_below_quorum" in res["errors"]


def test_v2_duplicate_signer_does_not_satisfy_quorum(tmp_path: Path) -> None:
    root, pairs = _repo_with_quorum(tmp_path, 1, 1)
    write_checkpoint(root, pairs)
    body = json.loads((root / "quality/checkpoint.json").read_text())
    # Replay the same signer a second time — still one distinct signer.
    body["signatures"].append(dict(body["signatures"][0]))
    (root / "quality/checkpoint.json").write_text(json.dumps(body))
    res = verify_checkpoint(root)
    assert res["ok"] is True  # threshold 1: a dup doesn't inflate the count
    # For a 2-of-2 registry the same replay must fail.
    root2, pairs2 = _repo_with_quorum(tmp_path / "r2", 2, 2)
    write_checkpoint(root2, pairs2)
    body2 = json.loads((root2 / "quality/checkpoint.json").read_text())
    body2["signatures"] = [body2["signatures"][0], dict(body2["signatures"][0])]
    (root2 / "quality/checkpoint.json").write_text(json.dumps(body2))
    res2 = verify_checkpoint(root2)
    assert res2["ok"] is False
    assert any(e.startswith("checkpoint_below_quorum") for e in res2["errors"])


def test_v2_forged_signer_and_tampered_payload(tmp_path: Path) -> None:
    root, pairs = _repo_with_quorum(tmp_path, 1, 1)
    write_checkpoint(root, pairs)
    body = json.loads((root / "quality/checkpoint.json").read_text())
    # A foreign key signs the same payload: unknown signer, quorum unmet.
    rogue_priv, rogue_pub = generate_keypair()
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from quant_fund.utils.hashing import canonical_json_bytes

    rogue_sig = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(rogue_priv)).sign(
        canonical_json_bytes(body["payload"])
    )
    from quant_fund.research.gate_signatures import key_id

    body["signatures"] = [{"key_id": key_id(rogue_pub), "signature": rogue_sig.hex()}]
    (root / "quality/checkpoint.json").write_text(json.dumps(body))
    res = verify_checkpoint(root)
    assert res["ok"] is False
    assert "signature_key_unknown" in res["errors"]

    # Payload tampering under a valid-registered signature still fails.
    write_checkpoint(root, pairs)
    body = json.loads((root / "quality/checkpoint.json").read_text())
    body["payload"]["heads"]["receipts/*.json"]["receipt"] = "corpus_epoch_evil.json"
    (root / "quality/checkpoint.json").write_text(json.dumps(body))
    res = verify_checkpoint(root)
    assert res["ok"] is False
    assert "signature_invalid" in res["errors"]


def test_malformed_registry_fails_closed(tmp_path: Path) -> None:
    root, pairs = _repo_with_quorum(tmp_path, 1, 1)
    write_checkpoint(root, pairs)
    (root / "quality/gate_quorum.json").write_text("{corrupt")
    res = verify_checkpoint(root)
    assert res["ok"] is False
    assert "quorum_registry_malformed" in res["errors"]
    assert any(e.startswith("checkpoint_below_quorum") for e in res["errors"])


def test_v2_spine_mixed_era(tmp_path: Path) -> None:
    """v1 archives + v2 head: the spine verifies each under its own era.

    Registry introduction requires key continuity: the gate key that signed
    the v1 parent must be a member of the first registry (an attacker-built
    set can't contain a key it can't sign under).
    """
    root, priv, pub = _repo(tmp_path)
    write_checkpoint(root, [(priv, pub)])  # v1 genesis — no registry yet
    from quant_fund.research.gate_signatures import init_quorum

    pairs = [(priv, pub)] + [generate_keypair()]
    init_quorum(root, [p for _s, p in pairs], threshold=2)
    write_checkpoint(root, pairs)  # v2 head over the v1 archive

    from quant_fund.research.checkpoint_chain import checkpoint_spine

    res = checkpoint_spine(root)
    assert res["ok"] is True
    assert res["spine_length"] == 2
