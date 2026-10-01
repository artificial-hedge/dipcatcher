"""run_audit lane: dip-bench runner contract pinned."""

from __future__ import annotations

from fx1.bench.run_audit import dip_run_audit, dip_run_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_runner_contract() -> None:
    r = dip_run_audit()
    assert r["events_found"] is True
    assert r["inputs_bound"] is True
    assert r["input_digests_64"] is True
    assert r["honesty_fields"] is True
    assert r["in_sample_labeled"] is True
    assert r["tamper_changes_pin"] is True
    assert r["empty_raises"] == "raise:FileNotFoundError"


def test_bench_ok_and_verifies() -> None:
    receipt = dip_run_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "dip_run_audit_test.json")
    assert result["valid"], result.get("errors")
