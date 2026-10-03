"""contamination_audit lane: contamination-evaluator contract pinned."""

from __future__ import annotations

from fx1.eval.contamination_audit import (
    contamination_audit,
    contamination_audit_bench,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_contamination_contract() -> None:
    r = contamination_audit()
    assert r["verbatim_flags"] is True
    assert r["pooled_does_not_overflag"] is True
    assert r["digest_newline_safe"] is True
    assert r["mink_inert"] is True
    assert r["gap_mismatch_raises"] == "raise:ValueError"
    assert r["gap_flags"] is True
    assert r["report_flagged"] is True
    assert r["half_input_skips_gap"] is True
    assert r["clean_report_unflagged"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = contamination_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "contamination_audit_test.json")
    assert result["valid"], result.get("errors")
