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

    bad_pin = tmp_path / "bad.json"
    bad_pin.write_text(
        json.dumps(
            {"heads": {key: {"receipt": epoch.name, "sha256": "0" * 64, "tree_root": "f" * 64}}}
        )
    )
    proc2 = _run("--absence", str(ap), "--pin", str(bad_pin), "--key", key)
    assert proc2.returncode == 1
    assert "merkle_root_not_pinned" in proc2.stdout
