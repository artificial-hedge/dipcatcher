"""run audit: manifest builder provenance + re-derived eval gate."""

from __future__ import annotations

from fx1.train.run_audit import run_audit, run_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_corpus_provenance() -> None:
    r = run_audit()
    assert r["corpus_stats"] is True
    assert r["no_provenance_refuses"] is True
    assert r["empty_refuses"] is True


def test_gate_rederived() -> None:
    r = run_audit()
    assert r["gate_ok"] is True
    assert r["flag_alone_fails"] is True
    assert r["no_flag_fails"] is True
    assert r["failing_honesty_fails"] is True
    assert r["domain_violation_fails"] is True


def test_manifest() -> None:
    r = run_audit()
    assert r["missing_corpus_fails"] is True
    assert r["missing_eval_fails"] is True
    assert r["binds_corpus"] is True
    assert r["binds_eval"] is True
    assert r["honesty_hardcoded"] is True
    assert r["written_matches"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = run_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "run_audit_test.json")
    assert result["valid"], result.get("errors")
