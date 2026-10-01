"""Tests for receipt_tombstone — the retraction record."""

from __future__ import annotations

import json
from pathlib import Path

from quant_fund.research.receipt_lattice import receipt_lattice
from quant_fund.research.receipt_tombstone import (
    load_tombstones,
    tombstone_body,
    tombstone_contract_errors,
    write_tombstone,
)


def _receipt(inputs: str, score: float) -> dict:
    return {
        "schema": "demo.v1",
        "inputs_sha256": inputs,
        "results": [{"head": "a", "pinball": score}],
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
    }


def _write(root: Path, name: str, doc: dict) -> Path:
    path = root / name
    path.write_text(json.dumps(doc))
    return path


def test_tombstone_retracts_conflicting_claim(tmp_path: Path) -> None:
    _write(tmp_path, "r1.json", _receipt("in-a", 0.42))
    bad = _write(tmp_path, "r2.json", _receipt("in-a", 0.43))
    assert receipt_lattice(tmp_path)["verdict"] == "inconsistent"

    tomb = write_tombstone(bad, corpus_dir=tmp_path, reason="superseded: miscomputed pinball")
    out = receipt_lattice(tmp_path)
    assert out["verdict"] == "consistent"
    assert out["n_retracted"] == 1
    assert out["retracted"]["r2.json"]["tombstone"] == tomb.name
    assert out["n_inconsistent_groups"] == 0


def test_tombstone_file_contributes_no_claims(tmp_path: Path) -> None:
    target = _write(tmp_path, "r1.json", _receipt("in-a", 0.42))
    tomb = write_tombstone(target, corpus_dir=tmp_path, reason="x")
    # The tombstone itself carries target fields that must never be claims.
    out = receipt_lattice(tmp_path)
    assert all("tombstone_" not in f for g in out["groups"] for f in g["files"])
    assert tomb.name in out["file_digests"]


def test_tombstone_target_drift_fails_closed(tmp_path: Path) -> None:
    target = _write(tmp_path, "r1.json", _receipt("in-a", 0.42))
    write_tombstone(target, corpus_dir=tmp_path, reason="x")
    target.write_text(json.dumps(_receipt("in-a", 0.99)))  # bytes drift after pin
    tombs = load_tombstones(tmp_path)
    assert tombs["active"] == {}
    assert any("tombstone_target_drift" in e for e in tombs["invalid"])
    out = receipt_lattice(tmp_path)
    assert any("tombstone_target_drift" in e["error"] for e in out["parse_errors"])


def test_unsealed_tombstone_is_invalid(tmp_path: Path) -> None:
    target = _write(tmp_path, "r1.json", _receipt("in-a", 0.42))
    body = tombstone_body(
        target_name="r1.json",
        target_sha256="0" * 64,
        reason="x",
    )
    _write(tmp_path, "tombstone_fake.json", body)  # never sealed
    tombs = load_tombstones(tmp_path)
    assert tombs["active"] == {}
    assert any("unsealed" in e for e in tombs["invalid"])
    assert target.is_file()


def test_contract_errors() -> None:
    good = tombstone_body(target_name="r1.json", target_sha256="ab" * 32, reason="superseded")
    assert tombstone_contract_errors(good) == []
    for key, val in (
        ("target_sha256", "zz" * 32),
        ("target_sha256", "short"),
        ("reason", ""),
        ("scope", []),
        ("live_pnl_claim", True),
    ):
        bad = dict(good)
        bad[key] = val
        assert tombstone_contract_errors(bad), key


def test_partial_scope_retracts_only_named_claims(tmp_path: Path) -> None:
    _write(tmp_path, "r1.json", _receipt("in-a", 0.42))
    bad = _write(tmp_path, "r2.json", _receipt("in-a", 0.43))
    write_tombstone(
        bad,
        corpus_dir=tmp_path,
        reason="only the pinball leaf is wrong",
        scope=["results[0].pinball"],
    )
    out = receipt_lattice(tmp_path)
    # pinball claim is retracted; head claim remains and agrees
    pinball_groups = [g for g in out["groups"] if "pinball" in g["claim_path"]]
    assert pinball_groups == []  # r2's pinball no longer joins the group
    assert out["verdict"] == "consistent"
    assert out["retracted"]["r2.json"]["scope"] == ["results[0].pinball"]


def test_retracted_findings_exit_the_fdr_pool(tmp_path: Path) -> None:
    from quant_fund.research.corpus_inference import corpus_audit

    _write(
        tmp_path,
        "r1.json",
        {**_receipt("in-a", 0.42), "kupiec_p": 0.001, "evidence_e": 50.0},
    )
    bad = _write(
        tmp_path,
        "r2.json",
        {**_receipt("in-a", 0.43), "kupiec_p": 0.0001, "evidence_e": 500.0},
    )
    before = corpus_audit(tmp_path)
    assert before["n_p_findings"] == 2

    write_tombstone(bad, corpus_dir=tmp_path, reason="superseded")
    after = corpus_audit(tmp_path)
    assert after["n_retracted"] == 1
    assert after["n_p_findings"] == 1  # r2's p-value no longer enters the pool
    assert all(f["source"] == "r1.json" for f in after["surviving_claims"])
