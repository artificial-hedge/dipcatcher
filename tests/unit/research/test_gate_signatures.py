"""Tests for gate_signatures — Ed25519 over the integrity pin manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    forged = b'{"forged": true}\n'
    (root / "quality/crown_jewels.json").write_bytes(forged)
    sig_body = json.loads((root / "gate_pins.sig").read_text())
    from quant_fund.utils.hashing import hash_bytes

    sig_body["payload"]["files"]["quality/crown_jewels.json"] = hash_bytes(forged)
    (root / "gate_pins.sig").write_text(json.dumps(sig_body, indent=2, sort_keys=True))
    res = verify_pin_signatures(root)
    assert res["ok"] is False
    assert res["errors"] == ["signature_invalid"]


def test_wrong_pubkey_fails(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    priv, pub = generate_keypair()
    _, other_pub = generate_keypair()
    sign_pins(root, priv, pub)
    (root / "quality/gate_signing.pub").write_text(other_pub + "\n")
    res = verify_pin_signatures(root)
    assert res["ok"] is False
    assert "key_id_mismatch" in res["errors"]


def test_legacy_sign_pins_refuses_mismatched_seed_without_writing(tmp_path: Path) -> None:
    import pytest

    root = _repo(tmp_path)
    private_hex, _public_hex = generate_keypair()
    _other_private_hex, other_public_hex = generate_keypair()
    with pytest.raises(ValueError, match="private seed does not match public key"):
        sign_pins(root, private_hex, other_public_hex)
    assert not (root / "quality/gate_signing.pub").exists()
    assert not (root / "gate_pins.sig").exists()


def test_legacy_sign_pins_refuses_replacing_committed_public_key(tmp_path: Path) -> None:
    import pytest

    root = _repo(tmp_path)
    private_hex, public_hex = generate_keypair()
    sign_pins(root, private_hex, public_hex)
    public_file = root / "quality/gate_signing.pub"
    signature_file = root / "gate_pins.sig"
    committed_public = public_file.read_bytes()
    committed_signature = signature_file.read_bytes()
    other_private_hex, other_public_hex = generate_keypair()
    with pytest.raises(ValueError, match="does not match committed public key"):
        sign_pins(root, other_private_hex, other_public_hex)
    assert public_file.read_bytes() == committed_public
    assert signature_file.read_bytes() == committed_signature


def test_legacy_sign_pins_reuses_committed_public_key(tmp_path: Path, monkeypatch) -> None:
    from quant_fund.utils import atomicio

    root = _repo(tmp_path)
    private_hex, public_hex = generate_keypair()
    sign_pins(root, private_hex, public_hex)
    public_file = root / "quality/gate_signing.pub"
    original_write = atomicio.atomic_write_text

    def write_without_replacing_public(path: Path, data: str) -> None:
        assert path != public_file
        original_write(path, data)

    monkeypatch.setattr(atomicio, "atomic_write_text", write_without_replacing_public)
    sign_pins(root, private_hex, public_hex)
    assert verify_pin_signatures(root)["ok"] is True


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


@pytest.mark.parametrize(
    "pub_hex",
    ["01" + "00" * 31, (2**255 - 18).to_bytes(32, "little").hex()],
)
def test_synthetic_identity_key_cannot_forge_quorum(tmp_path: Path, pub_hex: str) -> None:
    from quant_fund.research.gate_signatures import (
        QUORUM_REGISTRY_SCHEMA,
        QUORUM_SIG_SCHEMA,
        manifest_bytes,
    )

    root = _repo(tmp_path)
    kid = key_id(pub_hex)
    registry = {
        "schema": QUORUM_REGISTRY_SCHEMA,
        "threshold": 1,
        "keys": [{"key_id": kid, "pubkey": pub_hex}],
    }
    (root / "quality/gate_quorum.json").write_text(json.dumps(registry))
    # R is the Ed25519 base point and S is one. The identity public point
    # accepts this signature for any payload without a private key.
    forged_signature = (b"\x58" + b"\x66" * 31 + b"\x01" + b"\x00" * 31).hex()
    body = {
        "schema": QUORUM_SIG_SCHEMA,
        "algorithm": "ed25519",
        "payload": json.loads(manifest_bytes(root)),
        "signatures": [{"key_id": kid, "signature": forged_signature}],
    }
    (root / "gate_pins.sig").write_text(json.dumps(body))
    assert verify_pin_signatures(root)["ok"] is False


@pytest.mark.parametrize(
    "pub_hex",
    ["01" + "00" * 31, (2**255 - 18).to_bytes(32, "little").hex()],
)
def test_synthetic_identity_key_cannot_enter_quorum_registry(tmp_path: Path, pub_hex: str) -> None:
    from quant_fund.research.gate_signatures import init_quorum

    root = _repo(tmp_path)
    with pytest.raises(ValueError, match="invalid|malformed"):
        init_quorum(root, [pub_hex], threshold=1)
    assert not (root / "quality/gate_quorum.json").exists()


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


def test_quorum_rejects_mismatched_seed_before_writing(tmp_path: Path) -> None:
    import pytest

    from quant_fund.research.gate_signatures import (
        sign_pins_quorum,
        validate_quorum_signers,
    )

    root, keys = _quorum_repo(tmp_path)
    wrong_pair = (keys[2][0], keys[1][1])
    with pytest.raises(ValueError, match="private seed does not match public key"):
        validate_quorum_signers(root, [keys[0], wrong_pair])
    with pytest.raises(ValueError, match="private seed does not match public key"):
        sign_pins_quorum(root, [keys[0], wrong_pair])
    assert not (root / "gate_pins.sig").exists()


def test_quorum_rejects_duplicate_signer_before_writing(tmp_path: Path) -> None:
    import pytest

    from quant_fund.research.gate_signatures import sign_pins_quorum

    root, keys = _quorum_repo(tmp_path)
    with pytest.raises(ValueError, match="duplicate signer"):
        sign_pins_quorum(root, [keys[0], keys[0]])
    assert not (root / "gate_pins.sig").exists()


def test_quorum_downgrade_to_v1_fails_closed(tmp_path: Path) -> None:
    """A lone-key signature cannot masquerade under a committed registry."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from quant_fund.research.gate_signatures import GATE_SIGNATURES_SCHEMA
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

    root, keys = _quorum_repo(tmp_path)
    private_hex, public_hex = keys[0]
    payload = {"files": {rel: hash_bytes((root / rel).read_bytes()) for rel in SIGNED_FILES}}
    signature = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_hex)).sign(
        canonical_json_bytes(payload)
    )
    (root / "gate_pins.sig").write_text(
        json.dumps(
            {
                "schema": GATE_SIGNATURES_SCHEMA,
                "algorithm": "ed25519",
                "key_id": key_id(public_hex),
                "payload": payload,
                "signature": signature.hex(),
            }
        )
    )
    res = verify_pin_signatures(root)
    assert not res["ok"]
    assert res["errors"] == ["quorum_sig_missing"]


def test_legacy_sign_pins_refuses_present_or_malformed_quorum(tmp_path: Path) -> None:
    import pytest

    root, keys = _quorum_repo(tmp_path)
    with pytest.raises(ValueError, match="quorum registry present"):
        sign_pins(root, *keys[0])
    assert not (root / "gate_pins.sig").exists()
    assert not (root / "quality/gate_signing.pub").exists()

    (root / "quality/gate_quorum.json").write_text("{corrupt")
    with pytest.raises(ValueError, match="quorum registry present"):
        sign_pins(root, *keys[0])
    assert not (root / "gate_pins.sig").exists()
    assert not (root / "quality/gate_signing.pub").exists()


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
