"""pipeline_audit lane: staged pipeline gates + DPO pair contract."""

from __future__ import annotations

from fx1.train.pipeline_audit import pipeline_audit, pipeline_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_stage_gates() -> None:
    r = pipeline_audit()
    assert r["order_enforced"] is True
    assert r["empty_corpus_fails"] is True
    assert r["quality_gate_passed"] is True
    assert r["stage_is_eval_base"] is True
    assert r["bad_base_refused"] is True
    assert r["eval_base_written"] is True
    assert r["receipt_before_ckpt"] is True
    assert r["bad_candidate_refused"] is True


def test_dpo_pairs() -> None:
    r = pipeline_audit()
    assert r["pairs_cover_baits"] is True
    assert r["chosen_clean"] is True
    assert r["rejected_violating"] is True
    assert r["pair_prompt_is_bait"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = pipeline_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "pipeline_audit_test.json")
    assert result["valid"], result.get("errors")
