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
    # e-value mean merge: single finding → mean = the value itself.
    assert rep["corpus_evalue"] == 30.0
    assert rep["corpus_reject_at_alpha"] is True
    # inputs carry no data_label — the corpus honestly reports UNKNOWN
    assert rep["data_label"] == "UNKNOWN"


def test_evalue_mean_merge_valid_under_dependence(tmp_path: Path) -> None:
    """Arithmetic mean of e-values is an e-value under arbitrary dependence
    (Vovk–Wang): mean(30, 0.5) = 15.25 — the product (15.0) is kept only
    as a labeled diagnostic since receipts cannot be assumed independent."""
    _write(tmp_path, "a.json", {"kind": "k", "epromotion_evalue": 30.0})
    _write(tmp_path, "b.json", {"kind": "k", "drift_evalue": 0.5})
    rep = corpus_audit(tmp_path, q=0.05)
    assert rep["corpus_evalue"] == 15.25
    assert rep["corpus_evalue_product_dependence_assuming"] == 15.0
    assert "evalue_mean_merge" in rep["evidence"]
    assert "evalue_product_merge" not in rep["evidence"]
    # mean 15.25 < 1/0.05 → no corpus rejection from dependent evidence
    assert rep["corpus_reject_at_alpha"] is False


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


def test_corpus_receipt_v2_round_trip(tmp_path: Path) -> None:
    """receipt_version=2 seals the corpus_inference.v1 body in the envelope."""
    from quant_fund.research.corpus_inference import write_corpus_receipt
    from quant_fund.research.receipt_v2 import verify_receipt_file

    _write(tmp_path, "a.json", {"kind": "k", "dm_p": 0.001})
    _write(tmp_path, "b.json", {"kind": "k", "t_p": 0.9})
    rep = corpus_audit(tmp_path, q=0.05)
    path = write_corpus_receipt(rep, tmp_path, receipt_version=2)
    payload = json.loads(path.read_text())
    assert payload["schema"] == "receipt.v2"
    assert payload["payload"]["kind"] == "corpus_inference.v1"
    assert payload["payload"]["inputs_sha256"] == rep["inputs_sha256"]
    assert verify_receipt_file(path)["valid"] is True


def test_dataset_sha256_tracks_corpus_bytes(tmp_path: Path) -> None:
    """Two audits of the same corpus share dataset_sha256; adding a file
    changes it."""
    _write(tmp_path, "a.json", {"kind": "k", "p": 0.01})
    rep1 = corpus_audit(tmp_path)
    rep2 = corpus_audit(tmp_path)
    d = rep1["dataset_sha256"]
    assert isinstance(d, str) and len(d) == 64
    assert rep2["dataset_sha256"] == d
    _write(tmp_path, "b.json", {"kind": "k", "p": 0.9})
    rep3 = corpus_audit(tmp_path)
    assert rep3["dataset_sha256"] != d
    assert rep3["inputs_sha256"] != rep1["inputs_sha256"]


def test_dataset_sha256_edges_suite_health_on_same_corpus(tmp_path: Path) -> None:
    """Cross-lane edge: corpus_inference and suite_health digest the same
    per-file bytes, so one directory yields the identical dataset_sha256."""
    from quant_fund.research.suite_health import suite_health

    _write(tmp_path, "a.json", {"kind": "k", "p": 0.01})
    _write(tmp_path, "b.json", {"kind": "x", "evalue": 3.0})
    _, health = suite_health(tmp_path)
    audit = corpus_audit(tmp_path)
    assert health["dataset_sha256"] == audit["dataset_sha256"]


def test_membership_pins_the_input_set(tmp_path: Path) -> None:
    """A pinned membership audits exactly those basenames — new files
    arriving in the dir are ignored, so a frozen pin replays identically."""
    _write(tmp_path, "a.json", {"kind": "k", "p": 0.01})
    _write(tmp_path, "b.json", {"kind": "k", "p": 0.9})
    rep = corpus_audit(tmp_path, members={"a.json"})
    assert rep["n_receipts"] == 1
    assert rep["n_p_findings"] == 1
    assert rep["n_parse_errors"] == 0
    # a later arrival does not perturb the pinned audit
    _write(tmp_path, "c.json", {"kind": "k", "p": 0.001})
    rep2 = corpus_audit(tmp_path, members={"a.json"})
    assert rep2["n_receipts"] == 1
    assert rep2["inputs_sha256"] == rep["inputs_sha256"]
    assert rep2["dataset_sha256"] == rep["dataset_sha256"]


def test_membership_missing_member_fails_closed(tmp_path: Path) -> None:
    """A pinned member absent from the dir is a recorded error — the
    audit never silently reports a subset as the pinned corpus."""
    _write(tmp_path, "a.json", {"kind": "k", "p": 0.01})
    rep = corpus_audit(tmp_path, members={"a.json", "ghost.json"})
    assert rep["n_parse_errors"] == 1
    assert rep["parse_errors"][0]["file"] == "ghost.json"
    assert rep["parse_errors"][0]["error"] == "member_missing_from_dir"
