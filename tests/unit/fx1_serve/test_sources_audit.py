"""sources_audit lane: ledger/trace/notebook corpus-source contract."""

from __future__ import annotations

from fx1.data.sources_audit import sources_audit, sources_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_claims_live_scan() -> None:
    r = sources_audit()
    assert r["live_nested_detected"] is True
    assert r["live_in_list_detected"] is True
    assert r["falsy_forms_clean"] is True
    # pinned caveat: plural key evades the exact-match token
    assert r["plural_evades"] is True


def test_ledger_examples() -> None:
    r = sources_audit()
    assert r["live_is_negative"] is True
    assert r["clean_is_positive"] is True
    assert r["broken_skipped"] is True
    assert r["digest_bound"] is True


def test_trace_admission() -> None:
    r = sources_audit()
    assert r["admit_clean"] is True
    assert r["admit_unverified_as_negative"] is True
    assert r["admit_violating_refused"] is True
    # pinned caveat: tool-role content is never scanned
    assert r["tool_result_not_scanned"] is True
    assert r["three_written"] is True
    assert r["unverified_marked_negative"] is True
    assert r["trace_digest_bound"] is True


def test_notebooks() -> None:
    r = sources_audit()
    assert r["notebook_sections"] is True
    assert r["notebook_digest"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = sources_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "sources_audit_test.json")
    assert result["valid"], result.get("errors")
