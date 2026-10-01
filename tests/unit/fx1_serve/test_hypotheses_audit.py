"""hypotheses_audit lane: trace admissibility + score allowlist pinned."""

from __future__ import annotations

from fx1.hypotheses_audit import hypotheses_audit, hypotheses_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_score_gate() -> None:
    r = hypotheses_audit()
    assert all(c["outcome"] == "raise:ValueError" for c in r["forbidden_cases"])
    assert r["evasion_rejected"] == "raise:ValueError"
    assert r["unknown_rejected"] == "raise:ValueError"
    assert r["allowed_accept"] is True  # incl. rank_ic post-fix
    assert r["nonfinite_rejected"] == "raise:ValueError/raise:ValueError"


def test_trace_lifecycle() -> None:
    r = hypotheses_audit()
    assert r["pending_inadmissible"] is True
    assert r["decided_admissible"] is True
    assert r["verdict_changes_id"] is True
    assert r["sft_shape"] is True
    assert r["sft_receipt_bound"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = hypotheses_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "hypotheses_audit_test.json")
    assert result["valid"], result.get("errors")
