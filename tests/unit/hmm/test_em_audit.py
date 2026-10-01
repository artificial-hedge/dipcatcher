"""em_audit lane: Baum–Welch contract pinned."""

from __future__ import annotations

from quant_fund.hmm.em_audit import em_audit, em_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_all_em_checks_pass() -> None:
    results = em_audit()
    assert results["monotone_linear"]["ok"] is True
    assert results["monotone_log"]["ok"] is True
    assert results["log_linear_consistent"]["ok"] is True
    assert results["seed_bitwise"]["ok"] is True
    assert results["dead_state_row_preserved"]["ok"] is True
    assert results["underflow_boundary"]["ok"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = em_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "em_audit_test.json")
    assert result["valid"], result.get("errors")
