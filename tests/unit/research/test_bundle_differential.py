"""Differential fuzz: the two independent bundle verifiers must never
disagree.

The library (``quant_fund.research.auditor_bundle.verify_bundle``) and the
standalone stdlib implementation (``scripts/verify_auditor_bundle.py``) were
written to the same verdict contract; a mutation that one accepts and the
other rejects is itself a verifier bug — a divergence finding, independent
of which side is right. This suite lands seeded adversarial mutations on a
real bundle and requires verdict agreement on every one.
"""

from __future__ import annotations

import base64
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
QUALITY = REPO_ROOT / "quality"
SCRIPT = REPO_ROOT / "scripts" / "verify_auditor_bundle.py"

requires_tree = pytest.mark.skipif(
    not (QUALITY / "checkpoint.json").is_file()
    or not list((QUALITY / "witness").glob("*.json"))
    or not SCRIPT.is_file(),
    reason="integrity substrate not committed",
)


def _members(doc: dict) -> list[str]:
    return sorted(doc["files"])


def _flip_member(doc: dict, name: str, rng: random.Random, *, fix_hash: bool) -> dict:
    raw = bytearray(base64.b64decode(doc["files"][name]))
    raw[rng.randrange(len(raw))] ^= 1 << rng.randrange(8)
    doc["files"][name] = base64.b64encode(bytes(raw)).decode()
    if fix_hash:
        import hashlib

        doc["files_sha256"][name] = hashlib.sha256(bytes(raw)).hexdigest()
    return doc


def _drop_member(doc: dict, name: str) -> dict:
    doc["files"].pop(name)
    doc["files_sha256"].pop(name, None)
    return doc


def _corrupt_b64(doc: dict, name: str) -> dict:
    doc["files"][name] = "!!!not-base64!!!"
    return doc


def _rename_member(doc: dict, name: str) -> dict:
    doc["files"][name + ".bak"] = doc["files"].pop(name)
    doc["files_sha256"][name + ".bak"] = doc["files_sha256"].pop(name)
    return doc


def _extra_member(doc: dict, rng: random.Random) -> dict:
    payload = base64.b64encode(f"extra-{rng.randrange(1 << 20)}".encode()).decode()
    import hashlib

    doc["files"]["quality/extra_member.json"] = payload
    doc["files_sha256"]["quality/extra_member.json"] = hashlib.sha256(
        base64.b64decode(payload)
    ).hexdigest()
    return doc


def _flip_leaf(doc: dict) -> dict:
    body = doc["witness_proof"]["rekor"]["body_b64"]
    raw = bytearray(base64.b64decode(body))
    raw[0] ^= 0x01
    doc["witness_proof"]["rekor"]["body_b64"] = base64.b64encode(bytes(raw)).decode()
    return doc


def _flip_inclusion_hash(doc: dict, rng: random.Random) -> dict:
    ip = doc["witness_proof"]["rekor"]["inclusion_proof"]
    i = rng.randrange(len(ip["hashes"]))
    h = bytearray.fromhex(ip["hashes"][i])
    h[0] ^= 0x01
    ip["hashes"][i] = h.hex()
    return doc


def _drop_spine_record(doc: dict, rng: random.Random) -> dict:
    spine = sorted(n for n in doc["files"] if n.startswith("quality/checkpoints/"))
    assert len(spine) >= 2, "differential spine test needs history"
    # drop an interior record — not the head, not genesis
    _drop_member(doc, spine[rng.randrange(1, len(spine) - 1)] if len(spine) > 2 else spine[1])
    return doc


def _drop_witness_proof(doc: dict, rng: random.Random) -> dict:
    proofs = sorted(n for n in doc["files"] if n.startswith("quality/witness/"))
    _drop_member(doc, proofs[rng.randrange(len(proofs))])
    return doc


def _key_swap(doc: dict) -> dict:
    # swap the two committed pubkeys — the witness key must not validate as
    # the gate key and vice versa
    a, b = "quality/gate_signing.pub", "quality/witness_signing.pub"
    doc["files"][a], doc["files"][b] = doc["files"][b], doc["files"][a]
    doc["files_sha256"][a], doc["files_sha256"][b] = (
        doc["files_sha256"][b],
        doc["files_sha256"][a],
    )
    return doc


def _key_order_permute(doc: dict, rng: random.Random) -> dict:
    items = list(doc["files"].items())
    rng.shuffle(items)
    doc["files"] = dict(items)
    return doc


def _registry_swap(doc: dict) -> dict:
    """Present a forged live registry (attacker key) in place of the
    committed one — before quorum-era continuity this self-verified."""
    import hashlib

    from quant_fund.research.gate_signatures import generate_keypair

    _, pub = generate_keypair()
    kid = hashlib.sha256(bytes.fromhex(pub)).hexdigest()[:16]
    forged = {
        "schema": "gate_quorum.v1",
        "threshold": 1,
        "keys": [{"key_id": kid, "pubkey": pub}],
    }
    raw = (json.dumps(forged, indent=2, sort_keys=True) + "\n").encode()
    rel = "quality/gate_quorum.json"
    doc["files"][rel] = base64.b64encode(raw).decode()
    doc["files_sha256"][rel] = hashlib.sha256(raw).hexdigest()
    return doc


_MUTATIONS = (
    "member_byte_flip",
    "member_byte_flip_rehashed",
    "member_drop",
    "member_corrupt_b64",
    "member_rename",
    "extra_member",
    "rekor_leaf_flip",
    "inclusion_hash_flip",
    "spine_record_drop",
    "witness_proof_drop",
    "pubkey_swap",
    "key_order_permute",
    "registry_swap",
)

_PASS_ONLY = {"key_order_permute"}  # benign: JSON object order is not semantic


def _script_ok(path: Path) -> bool:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(path), "--offline"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode == 0


@requires_tree
def test_bundle_differential_verdict_agreement(tmp_path: Path) -> None:
    from quant_fund.research.auditor_bundle import build_bundle, verify_bundle

    base = json.loads(build_bundle(REPO_ROOT, tmp_path / "bundle.json").read_text())
    divergences: list[str] = []
    for seed in range(3):
        rng = random.Random(seed)
        for name in _MUTATIONS:
            doc = json.loads(json.dumps(base))  # deep copy
            m = _members(doc)
            if name == "member_byte_flip":
                doc = _flip_member(doc, rng.choice(m), rng, fix_hash=False)
            elif name == "member_byte_flip_rehashed":
                doc = _flip_member(doc, rng.choice(m), rng, fix_hash=True)
            elif name == "member_drop":
                doc = _drop_member(doc, rng.choice(m))
            elif name == "member_corrupt_b64":
                doc = _corrupt_b64(doc, rng.choice(m))
            elif name == "member_rename":
                doc = _rename_member(doc, rng.choice(m))
            elif name == "extra_member":
                doc = _extra_member(doc, rng)
            elif name == "rekor_leaf_flip":
                doc = _flip_leaf(doc)
            elif name == "inclusion_hash_flip":
                doc = _flip_inclusion_hash(doc, rng)
            elif name == "spine_record_drop":
                doc = _drop_spine_record(doc, rng)
            elif name == "witness_proof_drop":
                doc = _drop_witness_proof(doc, rng)
            elif name == "pubkey_swap":
                doc = _key_swap(doc)
            elif name == "key_order_permute":
                doc = _key_order_permute(doc, rng)
            elif name == "registry_swap":
                doc = _registry_swap(doc)

            mutant = tmp_path / f"mut_{seed}_{name}.json"
            mutant.write_text(json.dumps(doc))
            lib_ok = bool(verify_bundle(mutant, rekor_url=None)["ok"])
            script_ok = _script_ok(mutant)
            expected = name in _PASS_ONLY
            if expected:
                if not (lib_ok and script_ok):
                    divergences.append(
                        f"{name} seed={seed}: benign mutation rejected "
                        f"(lib={lib_ok} script={script_ok})"
                    )
            elif lib_ok != script_ok:
                divergences.append(f"{name} seed={seed}: DIVERGENT lib={lib_ok} script={script_ok}")
    assert divergences == [], "\n".join(divergences)
