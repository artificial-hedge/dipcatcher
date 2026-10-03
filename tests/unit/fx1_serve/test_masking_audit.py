"""masking_audit lane: masked-twin contract + escape surfaces pinned."""

from __future__ import annotations

from fx1.eval.masking_audit import masking_audit, masking_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_masking_contract() -> None:
    r = masking_audit()
    assert r["deterministic"] is True
    assert r["consistent_surface"] is True
    assert all(r["dates_masked"])
    assert r["allowlist_survives"] is True
    assert r["twin_required_masked"] is True
    assert r["interleaved"] is True


def test_escape_surfaces_flagged() -> None:
    r = masking_audit()
    # pinned findings — >6-char tickers and lowercase forms evade masking
    assert r["long_ticker_escapes"] is True
    assert r["lowercase_escapes"] is True
    assert r["gap_mismatch_raises"] == "raise:ValueError"


def test_bench_ok_and_verifies() -> None:
    receipt = masking_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "masking_audit_test.json")
    assert result["valid"], result.get("errors")
