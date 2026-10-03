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


def _quorum_repo(tmp_path: Path, n: int = 3, threshold: int = 2):
    from quant_fund.research.gate_signatures import init_quorum

    root = _repo(tmp_path)
    keys = [generate_keypair() for _ in range(n)]
    init_quorum(root, [pub for _, pub in keys], threshold=threshold)
    return root, keys


def test_quorum_round_trip(tmp_path: Path) -> None:
    from quant_fund.research.gate_signatures import sign_pins_quorum

    root, keys = _quorum_repo(tmp_path)
    sign_pins_quorum(root, keys[:2])
    res = verify_pin_signatures(root)
    assert res["ok"] and res["signed"] and res["quorum"] == "2/2"


def test_quorum_met_with_surplus_sig(tmp_path: Path) -> None:
    from quant_fund.research.gate_signatures import sign_pins_quorum

    root, keys = _quorum_repo(tmp_path)
    sign_pins_quorum(root, keys)  # all 3 sign, need 2
    res = verify_pin_signatures(root)
    assert res["ok"] and res["quorum"] == "3/2"


def test_quorum_not_met(tmp_path: Path) -> None:
    from quant_fund.research.gate_signatures import sign_pins_quorum

    root, keys = _quorum_repo(tmp_path)
    sign_pins_quorum(root, keys)
    body = json.loads((root / "gate_pins.sig").read_text())
    body["signatures"] = body["signatures"][:1]
    (root / "gate_pins.sig").write_text(json.dumps(body))
    res = verify_pin_signatures(root)
    assert not res["ok"]
    assert any(e.startswith("quorum_not_met") for e in res["errors"])


def test_quorum_emission_fails_closed(tmp_path: Path) -> None:
    import pytest

    from quant_fund.research.gate_signatures import sign_pins_quorum

    root, keys = _quorum_repo(tmp_path)
    with pytest.raises(ValueError, match="unattainable"):
        sign_pins_quorum(root, keys[:1])  # 1 signer < threshold 2
    assert not (root / "gate_pins.sig").exists()


def test_quorum_rejects_unregistered_signer(tmp_path: Path) -> None:
    import pytest

    from quant_fund.research.gate_signatures import sign_pins_quorum

    root, keys = _quorum_repo(tmp_path)
    outsider = generate_keypair()
    with pytest.raises(ValueError, match="not in quorum registry"):
        sign_pins_quorum(root, [*keys[:1], outsider])


def test_quorum_downgrade_to_v1_fails_closed(tmp_path: Path) -> None:
    """A lone-key signature cannot masquerade under a committed registry."""
    root, keys = _quorum_repo(tmp_path)
    sign_pins(root, *keys[0])  # writes v1 file + gate_signing.pub
    res = verify_pin_signatures(root)
    assert not res["ok"]
    assert res["errors"] == ["quorum_sig_missing"]


def test_quorum_unknown_signer_entry(tmp_path: Path) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from quant_fund.research.gate_signatures import (
        QUORUM_SIG_SCHEMA,
        manifest_bytes,
        sign_pins_quorum,
    )

    root, keys = _quorum_repo(tmp_path)
    sign_pins_quorum(root, keys[:2])
    body = json.loads((root / "gate_pins.sig").read_text())
    outsider = generate_keypair()
    forged_sig = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(outsider[0])).sign(
        manifest_bytes(root)
    )
    body["signatures"].append({"key_id": key_id(outsider[1]), "signature": forged_sig.hex()})
    assert body["schema"] == QUORUM_SIG_SCHEMA
    (root / "gate_pins.sig").write_text(json.dumps(body))
    res = verify_pin_signatures(root)
    assert not res["ok"]
    assert any(e.startswith("unknown_signer") for e in res["errors"])


def test_quorum_registry_malformed_fails_closed(tmp_path: Path) -> None:
    root, _keys = _quorum_repo(tmp_path)
    (root / "quality/gate_quorum.json").write_text('{"schema": "nope"}')
    res = verify_pin_signatures(root)
    assert not res["ok"]
    assert res["errors"] == ["registry_malformed"]


def test_quorum_duplicate_sig_counts_once(tmp_path: Path) -> None:
    from quant_fund.research.gate_signatures import sign_pins_quorum

    root, keys = _quorum_repo(tmp_path)
    sign_pins_quorum(root, keys[:2])
    body = json.loads((root / "gate_pins.sig").read_text())
    body["signatures"] = [body["signatures"][0], dict(body["signatures"][0])]
    (root / "gate_pins.sig").write_text(json.dumps(body))
    res = verify_pin_signatures(root)
    assert not res["ok"]
    assert any(e.startswith("duplicate_signer") for e in res["errors"])
    assert any(e.startswith("quorum_not_met") for e in res["errors"])
