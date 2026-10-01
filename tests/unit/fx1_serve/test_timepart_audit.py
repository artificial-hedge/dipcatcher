"""timepart_audit lane: post-cutoff partition contract pinned."""

from __future__ import annotations

from fx1.eval.timepart_audit import timepart_audit, timepart_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_partition_contract() -> None:
    r = timepart_audit()
    assert r["boundary_is_pre"] is True
    assert r["post_membership"] == ["garbage", "post"]
    assert r["undated_membership"] == ["undated"]
    assert r["malformed_sorts_post"] is True  # pinned caveat
    assert r["empty_post_none"] is True
    assert r["missing_result_fails"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = timepart_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "timepart_audit_test.json")
    assert result["valid"], result.get("errors")
