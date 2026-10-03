"""receipts_audit lane: eligibility matrix + coercion caveat pinned."""

from __future__ import annotations

from fx1.data.receipts_audit import receipts_audit, receipts_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_eligibility_matrix() -> None:
    r = receipts_audit()
    assert r["all_parseable_loaded"] is True
    assert r["fails_closed_default"] is True
    assert r["claim_contract_eligible"] is True
    assert r["explicit_live_wins"] is True
    assert r["synthetic_class"] is True
    assert r["nested_scanned"] is True
    assert r["digest_binds_bytes"] is True
    assert r["multi_dir"] is True


def test_coercion_flag() -> None:
    # coercion is closed: {research_only: "yes", live_pnl_claim: null}
    # is ineligible — research_only must be the literal true
    assert receipts_audit()["coercion_closed"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = receipts_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "receipts_audit_test.json")
    assert result["valid"], result.get("errors")
