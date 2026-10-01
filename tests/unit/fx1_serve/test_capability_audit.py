"""capability audit: aggregate gate semantics + rephrased twins."""

from __future__ import annotations

from fx1.eval.capability_audit import capability_audit, capability_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_rephrased() -> None:
    r = capability_audit()
    assert r["orphan_refuses"] is True
    assert r["covers_all"] is True
    assert r["twins_inherit"] is True
    assert r["gap_memorizer_flags"] is True
    assert r["gap_value_reflects"] is True
    assert r["memorizer_may_flag"] is True


def test_capability_gates() -> None:
    r = capability_audit()
    assert r["violating_closes_honesty"] is True
    assert r["violating_fails"] is True
    assert r["subgates_visible"] is True
    assert r["passed_implies_gates"] is True
    assert r["deterministic"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = capability_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "capability_audit_test.json")
    assert result["valid"], result.get("errors")
