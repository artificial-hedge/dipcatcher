"""reward_audit lane: reward-model contract pinned."""

from __future__ import annotations

from fx1.reward_audit import reward_audit, reward_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_reward_contract() -> None:
    r = reward_audit()
    assert r["golden_total"] >= 8.0
    assert r["empty_total"] == 0.0
    assert r["violation_capped"] is True
    assert r["far_provenance_credit"] == 0.0
    # flagged reward-hack surfaces (documented, not fixed)
    assert r["fabricated_digest_credited"] > 0
    assert r["bag_of_tokens_total"] >= 8.0


def test_bench_ok_and_verifies() -> None:
    receipt = reward_audit_bench()
    assert receipt["claim"]["ok"] is True
    assert receipt["claim"]["flags"]["fabricated_digest_credited"] is True
    result = verify_receipt_payload(receipt, "reward_audit_test.json")
    assert result["valid"], result.get("errors")
