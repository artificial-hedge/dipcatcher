"""custody: one bundle proves member→epoch→chain→pins→checkpoint→witness."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from quant_fund.research.custody import (
    custody_contract_errors,
    custody_proof,
    verify_custody_bundle,
)


def _seal(body: dict) -> dict:
    from quant_fund.research.receipt_v2 import seal_receipt

    return seal_receipt(body)


def _fixture(tmp_path: Path) -> Path:
    """Repo: receipts corpus (3 epochs), signed pins, checkpoint, pubkey."""

    from quant_fund.research.corpus_epoch import (
        corpus_epoch,
        update_heads_pin,
        write_epoch_receipt,
    )
    from quant_fund.research.crown_jewels import write_crown_jewels_pin
    from quant_fund.research.gate_signatures import generate_keypair, sign_pins
    from quant_fund.research.integrity_checkpoint import write_checkpoint

    root = tmp_path / "repo"
    (root / "receipts").mkdir(parents=True)
    (root / "quality").mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)

    from quant_fund.research.crown_jewels import DEFAULT_JEWELS

    for rel in DEFAULT_JEWELS:
        if rel == "quality/gate_signing.pub":
            continue  # written with the real pubkey below
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(f"jewel:{rel}\n".encode())

    pin = root / "quality/epoch_heads.json"
    pin.write_text(json.dumps({"schema": "epoch_heads.v1", "heads": {}}))

    def _receipt(name: str, val: float) -> None:
        body = {
            "kind": "synthetic_fixture.v1",
            "schema": "synthetic_fixture.v1",
            "research_only": True,
            "live_pnl_claim": False,
            "data_label": "SYNTHETIC",
            "inputs_sha256": "ab" * 32,
            "params": {},
            "results": [{"qlike": val}],
        }
        (root / "receipts" / name).write_text(json.dumps(_seal(body)))

    priv, pub = generate_keypair()
    (root / "quality/gate_signing.pub").write_text(pub)
    (root / ".fixture_priv").write_text(priv)

    # Three epochs, each stamping one more member than the last.
    for name, val in (("a.json", 1.0), ("b.json", 2.0), ("c.json", 3.0)):
        _receipt(name, val)
        ep = write_epoch_receipt(corpus_epoch(root / "receipts"), root / "receipts")
        update_heads_pin(pin, "receipts", "*.json", ep)
    write_crown_jewels_pin(root)
    ep = write_epoch_receipt(corpus_epoch(root / "quality"), root / "quality")
    update_heads_pin(pin, "quality", "*.json", ep)
    sign_pins(root, priv, pub)
    write_checkpoint(root, [(priv, pub)])
    return root


def test_custody_bundle_verifies_zero_repo(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    bundle = custody_proof("b.json", root / "receipts", pattern="*.json", root=root)
    assert bundle["kind"] == "custody_proof.v1"
    # b.json landed in epoch 2 — first_epoch names it, head is epoch 3.
    member_bytes = (root / "receipts/b.json").read_bytes()
    res = verify_custody_bundle(bundle, member_bytes)
    assert res["ok"], res["errors"]
    assert res["layers"]["inclusion"]["ok"]
    assert res["layers"]["signature"]["signed"] is True
    assert res["layers"]["checkpoint"]["ok"] is True


def test_custody_first_epoch_is_earliest_containing(tmp_path: Path) -> None:
    from quant_fund.research.custody import _chain_order
    from quant_fund.research.epoch_consistency import chain_index

    root = _fixture(tmp_path)
    index = chain_index(root / "receipts", pattern="*.json")
    order = _chain_order(index)
    bundle = custody_proof("a.json", root / "receipts", pattern="*.json", root=root)
    assert bundle["first_epoch"] == order[0]
    assert bundle["chain_head"] == order[-1]
    assert bundle["n_hops"] == 3


def test_custody_wrong_bytes_fail(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    bundle = custody_proof("b.json", root / "receipts", pattern="*.json", root=root)
    res = verify_custody_bundle(bundle, (root / "receipts/a.json").read_bytes())
    assert not res["ok"]
    assert "member_bytes_mismatch" in res["errors"]


def test_custody_tampered_hop_fails(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    bundle = custody_proof("a.json", root / "receipts", pattern="*.json", root=root)
    # Flip a byte inside hop 2's embedded file bytes.
    hop = bundle["hops"][1]
    raw = bytearray(__import__("base64").b64decode(hop["bytes_b64"]))
    raw[10] ^= 0xFF
    import base64

    bundle["hops"][1] = {**hop, "bytes_b64": base64.b64encode(bytes(raw)).decode()}
    res = verify_custody_bundle(bundle, (root / "receipts/a.json").read_bytes())
    assert not res["ok"]


def test_custody_unstamped_member_fails_closed(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    (root / "receipts/new.json").write_text(json.dumps({"kind": "x"}))
    with pytest.raises(ValueError, match="not stamped"):
        custody_proof("new.json", root / "receipts", pattern="*.json", root=root)


def test_custody_contract_clean(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    bundle = custody_proof("c.json", root / "receipts", pattern="*.json", root=root)
    assert custody_contract_errors(bundle) == []
    forged = dict(bundle, chain_head="corpus_epoch_ffffffffffffffff.json")
    assert "chain_head_mismatch" in custody_contract_errors(forged)


def test_member_timeline_tracks_byte_evolution(tmp_path: Path) -> None:
    """A member's committed digests across epochs are enumerated in order."""
    from quant_fund.research.corpus_epoch import (
        corpus_epoch,
        update_heads_pin,
        write_epoch_receipt,
    )
    from quant_fund.research.custody import member_timeline

    root = _fixture(tmp_path)
    before = member_timeline("b.json", root / "receipts", pattern="*.json")
    assert len(before) == 2  # stamped in epochs 2 and 3, same bytes
    assert before[0]["sha256"] == before[1]["sha256"]

    # Mutate the member and stamp a fourth epoch — lineage records the flip.
    doc = json.loads((root / "receipts/b.json").read_text())
    doc["results"] = [{"qlike": 9.0}]
    from quant_fund.research.receipt_v2 import seal_receipt

    (root / "receipts/b.json").write_text(json.dumps(seal_receipt(doc)))
    pin = root / "quality/epoch_heads.json"
    ep = write_epoch_receipt(corpus_epoch(root / "receipts"), root / "receipts")
    update_heads_pin(pin, "receipts", "*.json", ep)

    # The pin file just changed — re-sign and re-checkpoint before custody.
    from quant_fund.research.gate_signatures import sign_pins
    from quant_fund.research.integrity_checkpoint import write_checkpoint

    priv = (root / ".fixture_priv").read_text()
    pub = (root / "quality/gate_signing.pub").read_text()
    sign_pins(root, priv, pub)
    write_checkpoint(root, [(priv, pub)])

    after = member_timeline("b.json", root / "receipts", pattern="*.json")
    assert len(after) == 3
    assert after[-1]["sha256"] != after[0]["sha256"]
    assert [e["epoch_index"] for e in after] == [1, 2, 3]

    # Custody of the *new* bytes picks the mutating epoch as first_epoch.
    bundle = custody_proof("b.json", root / "receipts", pattern="*.json", root=root)
    assert bundle["first_epoch"] == after[-1]["epoch"]
    res = verify_custody_bundle(bundle, (root / "receipts/b.json").read_bytes())
    assert res["ok"], res["errors"]


def test_release_attestation_round_trip(tmp_path: Path) -> None:
    """Signed attestation binds artifact bytes to the pinned state."""
    from quant_fund.research.release_attestation import (
        release_attestation,
        release_contract_errors,
        verify_release_attestation,
    )

    root = _fixture(tmp_path)
    priv = (root / ".fixture_priv").read_text()
    pub = (root / "quality/gate_signing.pub").read_text().strip()
    wheel = b"fake-wheel-bytes"
    att = release_attestation(
        {"dipcatcher-0.1.0-py3-none-any.whl": wheel},
        root=root,
        private_seed_hex=priv,
        pubkey_hex=pub,
    )
    assert release_contract_errors(att) == []
    res = verify_release_attestation(att, {"dipcatcher-0.1.0-py3-none-any.whl": wheel}, root=root)
    assert res["ok"], res["errors"]

    # Tampered bytes fail; foreign key fails; stale state is flagged.
    bad = verify_release_attestation(att, {"dipcatcher-0.1.0-py3-none-any.whl": b"x"}, root=root)
    assert (
        not bad["ok"]
        and "artifact_digest_mismatch:dipcatcher-0.1.0-py3-none-any.whl" in bad["errors"]
    )

    from quant_fund.research.gate_signatures import generate_keypair

    _fpriv, fpub = generate_keypair()
    forged = release_attestation({"w": wheel}, root=root, private_seed_hex=_fpriv, pubkey_hex=fpub)
    fres = verify_release_attestation(forged, {"w": wheel}, root=root)
    assert not fres["ok"] and "key_id_mismatch" in fres["errors"]

    # Drift the pin file post-attestation → attestation_stale.
    pin = root / "quality/epoch_heads.json"
    pin.write_text(pin.read_text() + " ")
    stale = verify_release_attestation(att, {"dipcatcher-0.1.0-py3-none-any.whl": wheel}, root=root)
    assert not stale["ok"]
    assert any(e.startswith("attestation_stale:") for e in stale["errors"])


def test_digest_custody_finds_referenced_digest(tmp_path: Path) -> None:
    """A digest embedded in a stamped member's bytes gets a full bundle."""
    from quant_fund.research.custody import digest_custody

    root = _fixture(tmp_path)
    # a.json's sealed body contains inputs_sha256 = "ab"*32 — ask custody
    # for it and the carrier should be receipts/a.json.
    digest = "ab" * 32
    bundle = digest_custody(digest, root=root)
    assert bundle["carrier"] == "receipts/a.json"
    assert bundle["subject_digest"] == digest
    res = verify_custody_bundle(bundle, (root / "receipts/a.json").read_bytes())
    assert res["ok"], res["errors"]
    assert res["layers"]["subject_digest"]["ok"]

    # A digest not present anywhere fails closed at emission.
    with pytest.raises(ValueError, match="not referenced"):
        digest_custody("00" * 32, root=root)

    # Carrier bytes that no longer contain the digest → subject_digest_absent
    # (b'{}' also fails member_bytes_mismatch — both errors are expected).
    res_bad = verify_custody_bundle(bundle, b"{}")
    assert not res_bad["ok"]
    assert "subject_digest_absent" in res_bad["errors"]

    # Contract flags malformed subject_digest.
    forged = dict(bundle, subject_digest="xyz")
    assert "subject_digest" in custody_contract_errors(forged)


def _fake_rekor_entry(target_bytes: bytes, witness_key_pem: bytes, rekor_key) -> dict:
    """A cryptographically coherent Rekor entry signed by a fake Rekor key."""
    import base64
    import hashlib

    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    wkey = serialization.load_pem_private_key(witness_key_pem, password=None)
    wsig = wkey.sign(target_bytes, ec.ECDSA(hashes.SHA256()))
    wpub = wkey.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    body = {
        "kind": "hashedrekord",
        "apiVersion": "0.0.1",
        "spec": {
            "data": {
                "hash": {
                    "algorithm": "sha256",
                    "value": hashlib.sha256(target_bytes).hexdigest(),
                }
            },
            "signature": {
                "content": base64.b64encode(wsig).decode(),
                "publicKey": {"content": base64.b64encode(wpub).decode()},
            },
        },
    }
    body_b64 = base64.b64encode(json.dumps(body).encode()).decode()
    leaf = hashlib.sha256(b"\x00" + json.dumps(body).encode()).digest()
    root_hash = leaf.hex()  # single-leaf tree: root == leaf
    integrated = 1_700_000_000
    canon = json.dumps(
        {"body": body_b64, "integratedTime": integrated, "logID": "fake-log", "logIndex": 0},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    set_sig = rekor_key.sign(canon, ec.ECDSA(hashes.SHA256()))
    note_payload = f"rekor.sigstore.dev - 1\n1\n{root_hash}"
    note_sig = rekor_key.sign((note_payload + "\n").encode(), ec.ECDSA(hashes.SHA256()))
    note = (
        note_payload
        + "\n\n— rekor.sigstore.dev "
        + base64.b64encode(b"\x00" * 4 + note_sig).decode()
    )
    uuid = "0" * 32 + leaf.hex()
    return {
        uuid: {
            "body": body_b64,
            "logID": "fake-log",
            "logIndex": 0,
            "integratedTime": integrated,
            "verification": {
                "signedEntryTimestamp": base64.b64encode(set_sig).decode(),
                "inclusionProof": {
                    "logIndex": 0,
                    "treeSize": 1,
                    "rootHash": root_hash,
                    "hashes": [],
                    "checkpoint": note,
                },
            },
        }
    }


def _witness_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Attestation + a mocked Rekor submission signed by a fake Rekor key."""
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    from quant_fund.research.release_attestation import release_attestation

    root = _fixture(tmp_path)
    priv = (root / ".fixture_priv").read_text()
    pub = (root / "quality/gate_signing.pub").read_text().strip()
    wheel = b"fake-wheel-bytes"
    att = release_attestation({"w.whl": wheel}, root=root, private_seed_hex=priv, pubkey_hex=pub)
    witness_pem = ec.generate_private_key(ec.SECP256R1()).private_bytes(
        Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
    )
    rekor_key = ec.generate_private_key(ec.SECP256R1())
    rekor_pub_pem = rekor_key.public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    )

    def _post(target_bytes: bytes, key_pem: bytes, _url: str) -> dict:
        return _fake_rekor_entry(target_bytes, key_pem, rekor_key)

    monkeypatch.setattr("quant_fund.research.integrity_witness._post_entry", _post)
    return root, pub, wheel, att, witness_pem, rekor_pub_pem


def test_release_witness_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedded Rekor proof verifies with zero repo access."""
    from quant_fund.research.release_attestation import (
        release_contract_errors,
        verify_release_attestation,
        witness_release,
    )

    _root, pub, wheel, att, witness_pem, rekor_pub_pem = _witness_fixture(tmp_path, monkeypatch)
    witnessed = witness_release(att, witness_key_pem=witness_pem, rekor_pubkey_pem=rekor_pub_pem)
    assert release_contract_errors(witnessed) == []
    res = verify_release_attestation(witnessed, {"w.whl": wheel}, pubkey_hex=pub)
    assert res["ok"], res["errors"]


def test_release_witness_body_tamper_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from quant_fund.research.release_attestation import (
        verify_release_attestation,
        witness_release,
    )

    _root, pub, wheel, att, witness_pem, rekor_pub_pem = _witness_fixture(tmp_path, monkeypatch)
    witnessed = witness_release(att, witness_key_pem=witness_pem, rekor_pubkey_pem=rekor_pub_pem)
    witnessed["witness"]["rekor"]["body_b64"] = "e30="  # "{}"
    res = verify_release_attestation(witnessed, {"w.whl": wheel}, pubkey_hex=pub)
    assert not res["ok"]
    assert any(e.startswith("witness:") for e in res["errors"])


def test_release_witness_key_mismatch_with_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Committed pin keys disagreeing with embedded keys are flagged."""
    from quant_fund.research.release_attestation import (
        verify_release_attestation,
        witness_release,
    )

    root, pub, wheel, att, witness_pem, rekor_pub_pem = _witness_fixture(tmp_path, monkeypatch)
    witnessed = witness_release(att, witness_key_pem=witness_pem, rekor_pubkey_pem=rekor_pub_pem)
    res = verify_release_attestation(witnessed, {"w.whl": wheel}, root=root)
    assert not res["ok"]
    assert "witness_key_mismatch" in res["errors"]


def test_release_witness_malformed_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from quant_fund.research.release_attestation import (
        release_contract_errors,
        verify_release_attestation,
    )

    root, pub, wheel, att, _pem, _rkpub = _witness_fixture(tmp_path, monkeypatch)
    bad = dict(att, witness="not-a-proof")
    assert "witness_malformed" in release_contract_errors(bad)
    res = verify_release_attestation(bad, {"w.whl": wheel}, root=root)
    assert not res["ok"]
    assert any("witness_malformed" in e for e in res["errors"])


def test_custody_schema_dispatches_in_verify_receipt(tmp_path: Path) -> None:
    """custody_proof.v1 payloads get the contract check under verify-receipt."""
    from quant_fund.research.receipt_v2 import verify_receipt_payload

    root = _fixture(tmp_path)
    bundle = custody_proof("a.json", root / "receipts", pattern="*.json", root=root)
    res = verify_receipt_payload(bundle)
    assert not any(e.startswith("custody_") for e in res["errors"])
    # A contract-violating bundle is flagged through the same path.
    forged = dict(bundle, chain_head="corpus_epoch_ffffffffffffffff.json")
    forged_res = verify_receipt_payload(forged)
    assert "chain_head_mismatch" in forged_res["errors"]
