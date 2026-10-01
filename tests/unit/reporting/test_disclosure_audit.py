"""disclosure_audit lane: synthetic labeling on every reporting surface."""

from __future__ import annotations

from quant_fund.reporting.disclosure_audit import (
    disclosure_audit,
    disclosure_audit_bench,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_every_surface_discloses() -> None:
    r = disclosure_audit()
    assert r["tearsheet"]["banner_on_synthetic"] is True
    assert r["tearsheet"]["no_banner_on_real"] is True
    assert r["regime_performance"]["banner_on_synthetic"] is True
    assert r["evidence_report_synthetic"]["marker_in_warnings"] is True
    assert r["evidence_report_synthetic"]["status_not_complete"] is True
    assert r["evidence_report_missing_provenance"]["warns"] is True
    assert r["evidence_report_clean"]["no_synthetic_marker"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = disclosure_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "disclosure_audit_test.json")
    assert result["valid"], result.get("errors")
