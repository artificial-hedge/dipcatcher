"""corpus_audit lane: SFT builder contract pinned."""

from __future__ import annotations

from fx1.data.corpus_audit import corpus_audit, corpus_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_builder_contract() -> None:
    r = corpus_audit()
    assert r["stats_account"] is True
    assert r["load_balance"] is True
    assert r["one_positive_one_negative"] is True
    assert r["provenance_carried"] is True
    assert r["synthetic_note"] is True
    assert r["negative_is_refusal"] is True
    assert r["violation_skipped_not_emitted"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = corpus_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "corpus_audit_test.json")
    assert result["valid"], result.get("errors")
