"""harness_audit lane: fx-1 tool boundary contract pinned."""

from __future__ import annotations

from fx1.harness_audit import harness_audit, harness_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_harness_contract() -> None:
    r = harness_audit()
    assert r["api_absent"] is True
    assert r["lab_absent"] is True
    assert r["unknown_raises"] == "raise:KeyError"
    assert r["outside_config_raises"] == "raise:ValueError"
    assert r["relative_escape_raises"] == "raise:ValueError"
    assert r["symlink_escape_raises"] == "raise:ValueError"
    assert r["extra_args_config_raises"] == "raise:ValueError"
    assert r["extra_args_config_eq_raises"] == "raise:ValueError"
    assert r["exit_code_propagates"] is True
    assert r["timeout_bound_raises"].startswith("raise:")


def test_bench_ok_and_verifies() -> None:
    receipt = harness_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "harness_audit_test.json")
    assert result["valid"], result.get("errors")
