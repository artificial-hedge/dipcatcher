"""Tests for gate_signatures — Ed25519 over the integrity pin manifest."""

from __future__ import annotations

import json
from pathlib import Path

from quant_fund.research.gate_signatures import (
    SIGNED_FILES,
    generate_keypair,
    key_id,
    sign_pins,
    verify_pin_signatures,
)


def _repo(tmp_path: Path) -> Path:
    q = tmp_path / "quality"
    q.mkdir(parents=True)
    for rel in SIGNED_FILES:
        (tmp_path / rel).write_text(f'{{"pin": "{rel}"}}\n')
    return tmp_path


def test_sign_verify_round_trip(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    priv, pub = generate_keypair()
    sign_pins(root, priv, pub)
    res = verify_pin_signatures(root)
    assert res == {"ok": True, "signed": True, "errors": []}


def test_unsigned_is_neutral_not_pass_or_fail(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    res = verify_pin_signatures(root)
    assert res == {"ok": True, "signed": False, "errors": []}


def test_pin_drift_fails(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    priv, pub = generate_keypair()
    sign_pins(root, priv, pub)
    (root / "quality/crown_jewels.json").write_text('{"forged": true}\n')
    res = verify_pin_signatures(root)
    assert res["ok"] is False
    assert "pin_drift:quality/crown_jewels.json" in res["errors"]


def test_signature_forgery_fails(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    priv, pub = generate_keypair()
    sign_pins(root, priv, pub)
    # Attacker rewrites the pin AND the declared digest, but cannot sign.
    (root / "quality/crown_jewels.json").write_text('{"forged": true}\n')
    sig_body = json.loads((root / "gate_pins.sig").read_text())
    from quant_fund.utils.hashing import hash_bytes

    sig_body["payload"]["files"]["quality/crown_jewels.json"] = hash_bytes(b'{"forged": true}\n')
    (root / "gate_pins.sig").write_text(json.dumps(sig_body, indent=2, sort_keys=True))
    res = verify_pin_signatures(root)
    assert res["ok"] is False
    assert res["errors"] == ["signature_invalid"]


def test_wrong_pubkey_fails(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    priv, _pub = generate_keypair()
    _, other_pub = generate_keypair()
    sign_pins(root, priv, other_pub)
    res = verify_pin_signatures(root)
    assert res["ok"] is False
    assert "signature_invalid" in res["errors"] or "key_id_mismatch" in res["errors"]


def test_sig_without_pubkey_fails(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    priv, pub = generate_keypair()
    sign_pins(root, priv, pub)
    (root / "quality/gate_signing.pub").unlink()
    res = verify_pin_signatures(root)
    assert res["ok"] is False
    assert res["errors"] == ["pubkey_missing"]


def test_key_id_stable(tmp_path: Path) -> None:
    _priv, pub = generate_keypair()
    assert key_id(pub) == key_id(pub)
    assert len(key_id(pub)) == 16
