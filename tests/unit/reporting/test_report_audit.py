"""Report-audit lane: determinism + hygiene + self-consistency."""

from __future__ import annotations

from quant_fund.reporting.report_audit import _BAD_LITERALS, report_audit, report_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_bad_literal_regex_hits_degenerate_strings() -> None:
    assert _BAD_LITERALS.findall("sharpe=nan vol=inf x=-inf")
    assert not _BAD_LITERALS.findall("inflationary regime")
    assert not _BAD_LITERALS.findall("information ratio")


def test_audit_ok_on_clean_equity() -> None:
    r = report_audit(seed=3, n=120)
    assert r["deterministic_markdown"] is True
    assert r["bad_literals"] == []
    assert r["summary_self_consistent"] is True


def test_bench_seals_and_verifies() -> None:
    receipt = report_audit_bench(seeds=(0,))
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "report_audit_test.json")
    assert result["valid"], result.get("errors")
