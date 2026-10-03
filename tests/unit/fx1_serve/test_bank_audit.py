"""bank_audit lane: eval-bank self-consistency + scorer contract pinned."""

from __future__ import annotations

from fx1.eval.bank_audit import bank_audit, bank_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_banks_self_consistent() -> None:
    r = bank_audit()
    assert r["dup_names"] == []
    assert r["bad_patterns"] == []
    assert r["unsatisfiable_tasks"] == []
    assert r["vacuous_tasks"] == []
    assert r["honesty_tasks"] >= 10


def test_gate_contract() -> None:
    r = bank_audit()
    assert r["substring_required_tokens"] is True  # pinned hack surface
    assert r["empty_honesty_gate_closed"] is True
    assert r["violation_recorded"] is True
    assert r["violating_general_closes_gate"] is True
    assert r["digest_order_sensitive"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = bank_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "bank_audit_test.json")
    assert result["valid"], result.get("errors")
