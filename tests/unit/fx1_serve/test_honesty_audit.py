"""honesty audit: the fx-1 output gate's own contract."""

from __future__ import annotations

from fx1.honesty_audit import honesty_audit, honesty_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_refusals() -> None:
    r = honesty_audit()
    assert r["headlines_refuse"] is True
    assert r["live_claims_refuse"] is True
    assert r["unlabeled_synthetic_refuses"] is True
    assert r["lowercase_label_insufficient"] is True


def test_allowed() -> None:
    r = honesty_audit()
    assert r["clean_passes"] is True
    assert r["labeled_synthetic_passes"] is True
    assert r["synthetic_commentary_passes"] is True
    assert r["numeric_nonpresenting_passes"] is True


def test_flagged_surfaces() -> None:
    r = honesty_audit()
    assert r["flag_spelled_number_evades"] is True
    assert r["flag_glued_prefix_evades"] is True
    assert r["error_is_value_error"] is True
    assert r["identity_return"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = honesty_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "honesty_audit_test.json")
    assert result["valid"], result.get("errors")
