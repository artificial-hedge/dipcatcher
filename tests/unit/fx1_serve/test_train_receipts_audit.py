"""train receipts audit: binding + fail-closed verify contract."""

from __future__ import annotations

from fx1.train.receipts_audit import (
    train_receipt_audit,
    train_receipt_audit_bench,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_binding() -> None:
    r = train_receipt_audit()
    assert r["honesty_hardcoded"] is True
    assert r["four_digests_bound"] is True
    assert r["env_names_only"] is True
    assert r["non_git_fails_closed"] is True
    assert r["verify_ok"] is True


def test_fail_closed() -> None:
    r = train_receipt_audit()
    assert r["tampered_corpus_fails"] is True
    assert r["missing_file_fails"] is True
    assert r["forged_live_claim_fails"] is True
    assert r["forged_research_fails"] is True
    assert r["malformed_raises"] == "raise"


def test_bench_ok_and_verifies() -> None:
    receipt = train_receipt_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "train_receipt_audit_test.json")
    assert result["valid"], result.get("errors")
