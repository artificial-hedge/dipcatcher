"""Tests for fuzz_drill — the two-sided metamorphic property."""

from __future__ import annotations

import json
from pathlib import Path

from quant_fund.research.fuzz_drill import (
    _mutations,
    fuzz_contract_errors,
    fuzz_drill,
    write_fuzz_receipt,
)
from quant_fund.research.gate_signatures import generate_keypair, sign_pins
from quant_fund.research.integrity_checkpoint import write_checkpoint


def _repo(tmp_path: Path) -> tuple[Path, str, str]:
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
    return tmp_path, priv, pub


def test_mutations_are_two_sided(tmp_path: Path) -> None:
    """The sampler must emit BOTH must-fail and must-pass mutations."""
    root, priv, pub = _repo(tmp_path)
    (root / "receipts").mkdir()
    (root / "receipts/x.json").write_text("{}\n")
    write_checkpoint(root, [(priv, pub)])
    import random

    muts = _mutations(root, random.Random(1))
    expects = {e for _, e, _ in muts}
    assert "fail" in expects and "ok" in expects


def test_baseline_dirty_fails_closed(tmp_path: Path) -> None:
    """A dirty tree can't prove anything — drill refuses rather than
    reporting miscalibrated coverage."""
    root, priv, _pub = _repo(tmp_path)
    (root / "receipts").mkdir()
    write_checkpoint(root, [(priv, _pub)])
    # dirty the baseline: corrupt a pin file without re-signing
    (root / "quality/epoch_heads.json").write_text('{"tampered": true}\n')
    res = fuzz_drill(root, seed=1)
    assert res["ok"] is False
    assert res["verdict"] == "baseline_dirty"
    assert res["baseline_errors"]


def test_contract_coherence() -> None:
    good = {"ok": True, "verdict": "calibrated", "n_escaped": 0, "n_false_positive": 0}
    assert fuzz_contract_errors(good) == []
    bad = dict(good)
    bad["n_escaped"] = 2
    assert "ok_with_escapes" in fuzz_contract_errors(bad)
    lying = {"ok": False, "verdict": "calibrated"}
    assert "calibrated_not_ok" in fuzz_contract_errors(lying)


def test_write_fuzz_receipt_seals(tmp_path: Path) -> None:
    payload = {
        "schema": "fuzz_drill.v1",
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "simulated_only": False,
        "ok": True,
        "seed": 3,
        "mutations": [{"mutation": "m", "expect": "fail", "outcome": "correct"}],
        "n_mutations": 1,
        "n_escaped": 0,
        "n_false_positive": 0,
        "verdict": "calibrated",
    }
    out = write_fuzz_receipt(payload, tmp_path)
    body = json.loads(out.read_text())
    assert body["receipt_sha256"]
    assert body["schema"] == "fuzz_drill.v1"


def _mini_receipt(tmp_path: Path) -> Path:
    """A receipts/ dir with one v1 sealed receipt carrying a checked claim."""
    import json

    from quant_fund.research.receipt_v2 import seal_receipt

    rdir = tmp_path / "receipts"
    rdir.mkdir()
    body = {
        "schema": "fleet_eval.v1",
        "kind": "fleet_eval",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "simulated_only": True,
        "n_rows": 2,
        "n_error_rows": 0,
        "results": [
            {"head": "a", "score": 1.0},
            {"head": "b", "score": 2.0},
        ],
    }
    sealed = seal_receipt(body)
    path = rdir / "fleet_eval_test.json"
    path.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    return tmp_path


def test_receipt_fuzz_catches_resealed_forgery(tmp_path: Path) -> None:
    from quant_fund.research.fuzz_drill import receipt_fuzz

    res = receipt_fuzz(_mini_receipt(tmp_path), seed=3)
    assert res["schema"] == "receipt_fuzz.v1"
    assert res["n_mutations"] >= 1
    # The fleet_eval contract re-derives n_rows — a forged count is caught.
    assert any(m["outcome"] == "correct" for m in res["mutations"])


def test_receipt_fuzz_skips_epoch_records(tmp_path: Path) -> None:

    rdir = tmp_path / "receipts"
    rdir.mkdir()
    (rdir / "corpus_epoch_abc.json").write_text('{"schema": "corpus_epoch.v1", "n_members": 1}')
    from quant_fund.research.fuzz_drill import receipt_fuzz

    res = receipt_fuzz(tmp_path, seed=1)
    assert res["n_mutations"] == 0
