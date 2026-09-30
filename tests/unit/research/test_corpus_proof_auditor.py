"""Differential pin: scripts/verify_corpus_proof.py ↔ epoch_merkle library.

The standalone script is the third-party oracle — a divergence between its
verdict and the library's is itself a finding. Covers a real emitted proof,
forged-index absence frames, and a stale-pin rejection.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "verify_corpus_proof.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_script_verifies_library_emitted_proof(tmp_path: Path) -> None:
    """Library-emitted sealed proof verifies clean under the script."""
    from quant_fund.research.corpus_epoch import corpus_epoch, epoch_heads_key, write_epoch_receipt
    from quant_fund.research.epoch_merkle import member_proof
    from quant_fund.research.receipt_v2 import seal_receipt

    corpus = tmp_path / "receipts"
    corpus.mkdir()
    for name in ("a.json", "c.json", "e.json"):
        (corpus / name).write_text(json.dumps({"v": name}))
    epoch = write_epoch_receipt(corpus_epoch(corpus), corpus)
    members = {m["name"]: m["sha256"] for m in json.loads(epoch.read_text())["members"]}
    from quant_fund.research.epoch_merkle import merkle_root

    pin = tmp_path / "pin.json"
    pin.write_text(
        json.dumps(
            {
                "heads": {
                    epoch_heads_key(corpus, "*.json"): {
                        "receipt": epoch.name,
                        "sha256": "0" * 64,
                        "tree_root": merkle_root(members),
                    }
                }
            }
        )
    )
    body = member_proof(corpus, "c.json")
    proof_path = tmp_path / "proof.json"
    proof_path.write_text(json.dumps(seal_receipt(body)))

    proc = _run("--proof", str(proof_path), "--pin", str(pin))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok" in proc.stdout


def test_script_rejects_forged_index(tmp_path: Path) -> None:
    """A proof with a fudged leaf_index must not recompute the pin."""
    from quant_fund.research.corpus_epoch import corpus_epoch, epoch_heads_key, write_epoch_receipt
    from quant_fund.research.epoch_merkle import member_proof, merkle_root

    corpus = tmp_path / "receipts"
    corpus.mkdir()
    for name in ("a.json", "c.json", "e.json"):
        (corpus / name).write_text(json.dumps({"v": name}))
    epoch = write_epoch_receipt(corpus_epoch(corpus), corpus)
    members = {m["name"]: m["sha256"] for m in json.loads(epoch.read_text())["members"]}
    pin = tmp_path / "pin.json"
    pin.write_text(
        json.dumps(
            {
                "heads": {
                    epoch_heads_key(corpus, "*.json"): {
                        "receipt": epoch.name,
                        "sha256": "0" * 64,
                        "tree_root": merkle_root(members),
                    }
                }
            }
        )
    )
    body = member_proof(corpus, "c.json")
    forged = dict(body, leaf_index=(body["leaf_index"] + 1) % body["n_members"])
    proof_path = tmp_path / "forged.json"
    proof_path.write_text(json.dumps(forged))
    proc = _run("--proof", str(proof_path), "--pin", str(pin))
    assert proc.returncode == 1
    assert "path_side_mismatch" in proc.stdout or "path_depth_mismatch" in proc.stdout


def test_script_absence_bracket(tmp_path: Path) -> None:
    """A real absence proof verifies; a wrong pin root is rejected."""
    from quant_fund.research.corpus_epoch import corpus_epoch, epoch_heads_key, write_epoch_receipt
    from quant_fund.research.epoch_merkle import absence_proof, merkle_root

    corpus = tmp_path / "receipts"
    corpus.mkdir()
    for name in ("a.json", "c.json", "e.json"):
        (corpus / name).write_text(json.dumps({"v": name}))
    epoch = write_epoch_receipt(corpus_epoch(corpus), corpus)
    members = {m["name"]: m["sha256"] for m in json.loads(epoch.read_text())["members"]}
    root = merkle_root(members)
    pin = tmp_path / "pin.json"
    key = epoch_heads_key(corpus, "*.json")
    pin.write_text(
        json.dumps({"heads": {key: {"receipt": epoch.name, "sha256": "0" * 64, "tree_root": root}}})
    )
    proof = absence_proof(members, "b.json")
    ap = tmp_path / "abs.json"
    ap.write_text(json.dumps(proof))
    proc = _run("--absence", str(ap), "--pin", str(pin), "--key", key)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # Edge bound on a promoted leaf (idx == n-1, odd n): the promotion level
    # contributes no path entry — a divergence once shipped here.
    edge = absence_proof(members, "z.json")
    ep = tmp_path / "abs_edge.json"
    ep.write_text(json.dumps(edge))
    proc_edge = _run("--absence", str(ep), "--pin", str(pin), "--key", key)
    assert proc_edge.returncode == 0, proc_edge.stdout + proc_edge.stderr

    bad_pin = tmp_path / "bad.json"
    bad_pin.write_text(
        json.dumps(
            {"heads": {key: {"receipt": epoch.name, "sha256": "0" * 64, "tree_root": "f" * 64}}}
        )
    )
    proc2 = _run("--absence", str(ap), "--pin", str(bad_pin), "--key", key)
    assert proc2.returncode == 1
    assert "merkle_root_not_pinned" in proc2.stdout


def _signed_checkpoint(pin_heads: dict, pins: dict, tmp_path: Path) -> tuple[Path, Path]:
    """A checkpoint envelope signed by a throwaway key — exercises the
    script's Ed25519 auth path end-to-end."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    key = Ed25519PrivateKey.generate()
    pub_hex = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    payload = {
        "schema": "integrity_checkpoint.v1",
        "at": "2026-01-01T00:00:00+00:00",
        "pins": pins,
        "heads": pin_heads,
        "missing_pins": [],
        "prev_sha256": None,
        "spine": {"n_archives": 0, "tip": None},
        "witness": {"n_proofs": 0, "proofs": {}},
        "code": {"revision": "test", "worktree_sha256": "0" * 64},
    }
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    sig = key.sign(canon).hex()
    cp = tmp_path / "checkpoint.json"
    cp.write_text(
        json.dumps(
            {
                "schema": "integrity_checkpoint_sig.v1",
                "algorithm": "ed25519",
                "key_id": pub_hex[:16],
                "payload": payload,
                "signature": sig,
            }
        )
    )
    pub = tmp_path / "gate_signing.pub"
    pub.write_text(pub_hex)
    return cp, pub


def test_script_checkpoint_mode(tmp_path: Path) -> None:
    """Proof + signed checkpoint alone (no pin file) verifies; a checkpoint
    signed by a foreign key or cross-pinning other bytes fails."""
    from quant_fund.research.corpus_epoch import corpus_epoch, epoch_heads_key, write_epoch_receipt
    from quant_fund.research.epoch_merkle import member_proof, merkle_root
    from quant_fund.research.receipt_v2 import seal_receipt

    corpus = tmp_path / "receipts"
    corpus.mkdir()
    for name in ("a.json", "c.json", "e.json"):
        (corpus / name).write_text(json.dumps({"v": name}))
    epoch = write_epoch_receipt(corpus_epoch(corpus), corpus)
    members = {m["name"]: m["sha256"] for m in json.loads(epoch.read_text())["members"]}
    root = merkle_root(members)
    key = epoch_heads_key(corpus, "*.json")
    pin_heads = {key: {"receipt": epoch.name, "sha256": "0" * 64, "tree_root": root}}

    pin_file = tmp_path / "quality" / "epoch_heads.json"
    pin_file.parent.mkdir(parents=True, exist_ok=True)
    pin_file.write_text(json.dumps({"heads": pin_heads}))
    pins = {"quality/epoch_heads.json": "0" * 64}
    import hashlib

    pins["quality/epoch_heads.json"] = hashlib.sha256(pin_file.read_bytes()).hexdigest()

    cp, pub = _signed_checkpoint(pin_heads, pins, tmp_path)
    body = member_proof(corpus, "c.json")
    proof_path = tmp_path / "proof.json"
    proof_path.write_text(json.dumps(seal_receipt(body)))

    # checkpoint-only mode
    proc = _run(
        "--proof",
        str(proof_path),
        "--checkpoint",
        str(cp),
        "--pubkey",
        str(pub),
        "--key",
        key,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # checkpoint + pin cross-anchor
    proc2 = _run(
        "--proof",
        str(proof_path),
        "--pin",
        str(pin_file),
        "--checkpoint",
        str(cp),
        "--pubkey",
        str(pub),
        "--key",
        key,
    )
    assert proc2.returncode == 0, proc2.stdout + proc2.stderr

    # swapped pin bytes must trip the cross-anchor
    pin_file.write_text(json.dumps({"heads": pin_heads, "pad": 1}))
    proc3 = _run(
        "--proof",
        str(proof_path),
        "--pin",
        str(pin_file),
        "--checkpoint",
        str(cp),
        "--pubkey",
        str(pub),
        "--key",
        key,
    )
    assert proc3.returncode == 1
    assert "checkpoint_pin_mismatch" in proc3.stdout

    # foreign-signed checkpoint must fail the signature layer
    (tmp_path / "f").mkdir()
    cp2, pub2 = _signed_checkpoint(pin_heads, pins, tmp_path / "f")
    forged = json.loads(cp2.read_text())
    forged["payload"]["heads"] = pin_heads  # same content, foreign key
    cp3 = tmp_path / "checkpoint_forged.json"
    cp3.write_text(json.dumps(forged))
    proc4 = _run(
        "--proof",
        str(proof_path),
        "--checkpoint",
        str(cp3),
        "--pubkey",
        str(pub),
        "--key",
        key,
    )
    assert proc4.returncode == 1
    assert "checkpoint_signature_invalid" in proc4.stdout
