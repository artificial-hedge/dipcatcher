"""doctor_audit lane: presence-flags-only contract pinned."""

from __future__ import annotations

from fx1.doctor_audit import doctor_audit, doctor_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_no_secret_leakage() -> None:
    r = doctor_audit()
    assert r["sentinel_absent"] is True
    assert r["flags_only"] is True
    assert r["unset_flags"] is True


def test_degraded_states() -> None:
    r = doctor_audit()
    assert r["missing_receipts"] is True
    assert r["empty_receipts_zero"] is True
    assert r["corrupt_ledger"] is True
    assert r["ledger_missing"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = doctor_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "doctor_audit_test.json")
    assert result["valid"], result.get("errors")
