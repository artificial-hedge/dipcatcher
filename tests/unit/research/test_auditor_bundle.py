"""auditor_bundle: zero-trust verification against the live Rekor record.

The honest-path tests need the committed quality/ tree (checkpoint, pins,
witness proof) and are skipped when the substrate isn't present — the same
skip pattern as test_integrity_witness.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
QUALITY = REPO_ROOT / "quality"

_REQUIRES = [
    QUALITY / "checkpoint.json",
    QUALITY / "epoch_heads.json",
    QUALITY / "crown_jewels.json",
    QUALITY / "gate_signing.pub",
    REPO_ROOT / "gate_pins.sig",
    QUALITY / "witness_signing.pub",
    QUALITY / "rekor_pubkey.pem",
]
requires_tree = pytest.mark.skipif(
    not all(p.is_file() for p in _REQUIRES) or not list((QUALITY / "witness").glob("*.json")),
    reason="integrity substrate not committed",
)


def _build(tmp_path: Path) -> Path:
    from quant_fund.research.auditor_bundle import build_bundle

    return build_bundle(REPO_ROOT, tmp_path / "bundle.json")


@requires_tree
def test_build_bundle_contains_all_members(tmp_path: Path) -> None:
    from quant_fund.research.auditor_bundle import BUNDLE_MEMBERS, BUNDLE_SCHEMA

    bundle = json.loads(_build(tmp_path).read_text())
    assert bundle["schema"] == BUNDLE_SCHEMA
    assert set(bundle["files"]) == set(BUNDLE_MEMBERS)
    assert bundle["witness_proof"]["schema"] == "integrity_witness.v1"
    # every member hash is declared
    assert set(bundle["files_sha256"]) == set(BUNDLE_MEMBERS)


@requires_tree
def test_verify_bundle_honest_pinned_key(tmp_path: Path) -> None:
    from quant_fund.research.auditor_bundle import verify_bundle

    path = _build(tmp_path)
    res = verify_bundle(path, rekor_url=None)
    assert res["ok"], res["errors"]
    assert res["log_index"]


@requires_tree
def test_verify_bundle_caller_pinned_key(tmp_path: Path) -> None:
    from quant_fund.research.auditor_bundle import verify_bundle

    res = verify_bundle(
        _build(tmp_path),
        rekor_pubkey_pem=(QUALITY / "rekor_pubkey.pem").read_bytes(),
    )
    assert res["ok"], res["errors"]


@requires_tree
def test_verify_bundle_tampered_checkpoint_fails(tmp_path: Path) -> None:
    """Mutate a member AND fix its declared digest: the deeper chain —
    checkpoint bytes must hash to the logged Rekor digest — must catch it."""
    from quant_fund.research.auditor_bundle import verify_bundle

    bundle = json.loads(_build(tmp_path).read_text())
    raw = bytearray(base64.b64decode(bundle["files"]["quality/checkpoint.json"]))
    raw[8] ^= 0x01
    mutated = bytes(raw)
    bundle["files"]["quality/checkpoint.json"] = base64.b64encode(mutated).decode()
    import hashlib

    bundle["files_sha256"]["quality/checkpoint.json"] = hashlib.sha256(mutated).hexdigest()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(bundle))
    res = verify_bundle(bad, rekor_url=None)
    assert not res["ok"]
    assert any("witness" in e or "checkpoint" in e for e in res["errors"])


@requires_tree
def test_verify_bundle_swapped_witness_key_fails(tmp_path: Path) -> None:
    """A bundle carrying a foreign witness pubkey must fail even when its
    own declared digests are patched — the Rekor entry records the real key."""
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    from quant_fund.research.auditor_bundle import verify_bundle

    bundle = json.loads(_build(tmp_path).read_text())
    evil = (
        ec.generate_private_key(ec.SECP256R1())
        .public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    )
    bundle["files"]["quality/witness_signing.pub"] = base64.b64encode(evil).decode()
    import hashlib

    bundle["files_sha256"]["quality/witness_signing.pub"] = hashlib.sha256(evil).hexdigest()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(bundle))
    res = verify_bundle(bad, rekor_url=None)
    assert not res["ok"]
    assert "witness_pubkey_diverges_from_log" in res["errors"]


@requires_tree
def test_verify_bundle_missing_member_fails(tmp_path: Path) -> None:
    from quant_fund.research.auditor_bundle import verify_bundle

    bundle = json.loads(_build(tmp_path).read_text())
    del bundle["files"]["quality/epoch_heads.json"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(bundle))
    res = verify_bundle(bad, rekor_url=None)
    assert not res["ok"]
    assert "missing:quality/epoch_heads.json" in res["errors"]


def test_verify_bundle_rejects_malformed(tmp_path: Path) -> None:
    from quant_fund.research.auditor_bundle import verify_bundle

    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert verify_bundle(bad)["errors"] == ["bundle_malformed"]
    bad.write_text(json.dumps({"schema": "x"}))
    assert verify_bundle(bad)["errors"] == ["schema_mismatch"]
    bad.write_text(json.dumps({"schema": "auditor_bundle.v1", "files": {}, "witness_proof": {}}))
    res = verify_bundle(bad)
    assert not res["ok"]
    assert res["errors"][0].startswith("missing:")


def test_bundle_contract_errors() -> None:
    from quant_fund.research.auditor_bundle import bundle_contract_errors

    assert bundle_contract_errors({}) != []
    ok_body = {
        "schema": "auditor_bundle.v1",
        "files": {
            rel: base64.b64encode(b"x").decode()
            for rel in (
                "quality/checkpoint.json",
                "quality/epoch_heads.json",
                "quality/crown_jewels.json",
                "quality/gate_signing.pub",
                "gate_pins.sig",
                "quality/witness_signing.pub",
                "quality/rekor_pubkey.pem",
            )
        },
        "witness_proof": {},
    }
    import hashlib

    ok_body["files_sha256"] = {rel: hashlib.sha256(b"x").hexdigest() for rel in ok_body["files"]}
    assert bundle_contract_errors(ok_body) == []
