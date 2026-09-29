"""corpus_inference: BH-FDR over committed receipt p-values + e-value merge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.research.corpus_inference import (
    _bh_survivors,
    corpus_audit,
    harvest_findings,
)


def _write(root: Path, name: str, doc: dict) -> None:
    (root / name).write_text(json.dumps(doc))


def test_harvest_tags_p_and_e_separately() -> None:
    doc = {
        "kind": "fleet.v1",
        "family": "discovery",
        "results": {"dm_p": 0.01, "epromotion_evalue": 25.0, "anytime_p": 0.04},
        "params": {"alpha": 0.05, "n_origins": 24, "q_level": 0.05},
    }
    fs = harvest_findings(doc, "r.json")
    stats = {f["path"]: f["stat"] for f in fs}
    assert stats["results.dm_p"] == "p"
    assert stats["results.anytime_p"] == "p"
    assert stats["results.epromotion_evalue"] == "e"
    # Thresholds and counts never harvested.
    assert not any("alpha" in f["path"] or "n_origins" in f["path"] for f in fs)


def test_bh_correct_math() -> None:
    # p = [0.001, 0.01, 0.4, 0.9] at q=0.05: crit=[0.0125,0.025,0.0375,0.05]
    # sorted: 0.001<=0.0125 ✓, 0.01<=0.025 ✓, 0.4>0.0375 ✗, 0.9>0.05 ✗ → 2 survive.
    flags = _bh_survivors([0.001, 0.01, 0.4, 0.9], 0.05)
    assert flags == [True, True, False, False]
    # Empty + single.
    assert _bh_survivors([], 0.05) == []
    assert _bh_survivors([0.9], 0.05) == [False]
    assert _bh_survivors([0.01], 0.05) == [True]


def test_corpus_audit_end_to_end(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "a.json",
        {"kind": "k", "dm_p": 0.001, "coverage_p": 0.4, "epromotion_evalue": 30.0},
    )
    _write(tmp_path, "b.json", {"kind": "k", "t_p": 0.9, "nested": {"x_p": 0.0004}})
    rep = corpus_audit(tmp_path, q=0.05)
    assert rep["kind"] == "corpus_inference.v1"
    assert rep["n_receipts"] == 2
    assert rep["n_parse_errors"] == 0
    assert rep["n_p_findings"] == 3
    assert rep["n_e_findings"] == 1
    # Two tiny p's + one 0.9: BH at q=0.05, m=3 → crit 0.0167/0.033/0.05.
    # sorted: 0.0004<=0.0167 ✓, 0.001<=0.033 ✓, 0.9>0.05 ✗ → 2 survive.
    assert rep["n_survivors"] == 2
    srcs = {c["source"] for c in rep["surviving_claims"]}
    assert srcs == {"a.json", "b.json"}
    # e-value product merge: 30.0 ≥ 1/0.05 → corpus rejects null.
    assert rep["corpus_evalue"] == 30.0
    assert rep["corpus_reject_at_alpha"] is True
    # inputs carry no data_label — the corpus honestly reports UNKNOWN
    assert rep["data_label"] == "UNKNOWN"


def test_parse_error_recorded(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{not json")
    _write(tmp_path, "ok.json", {"kind": "k", "p": 0.01})
    rep = corpus_audit(tmp_path)
    assert rep["n_parse_errors"] == 1
    assert rep["parse_errors"][0]["file"] == "bad.json"
    assert rep["n_p_findings"] == 1


def test_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        corpus_audit(tmp_path / "nope")
    with pytest.raises(ValueError, match="q"):
        corpus_audit(tmp_path, q=1.5)


def test_real_receipts_dir_parses() -> None:
    root = Path(__file__).resolve().parents[3] / "receipts"
    if not root.is_dir():
        pytest.skip("no committed receipts")
    rep = corpus_audit(root, glob="*.json")
    assert rep["n_receipts"] >= 1
    assert rep["n_parse_errors"] == 0


def test_corpus_label_derived_from_inputs(tmp_path: Path) -> None:
    """Corpus receipts inherit the input labels; mixed corpora say MIXED."""
    import json

    for name, label in (("a", "yahoo_eod"), ("b", "yahoo_eod")):
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"kind": "x", "data_label": label, "p_value": 0.01})
        )
    rep = corpus_audit(tmp_path)
    assert rep["data_label"] == "yahoo_eod"
    assert rep["params"]["input_labels"] == {"a.json": "yahoo_eod", "b.json": "yahoo_eod"}

    (tmp_path / "c.json").write_text(
        json.dumps({"kind": "x", "data_label": "stooq", "p_value": 0.02})
    )
    rep2 = corpus_audit(tmp_path)
    assert rep2["data_label"] == "MIXED"
