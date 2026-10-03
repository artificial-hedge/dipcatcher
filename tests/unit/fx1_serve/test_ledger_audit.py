"""ledger_audit lane: corpus hash-chain contract + caveat pinned."""

from __future__ import annotations

from fx1.data.ledger_audit import ledger_audit, ledger_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_chain_integrity() -> None:
    r = ledger_audit()
    assert r["fresh_valid"] is True
    assert r["genesis_link"] is True
    assert r["indexes_dense"] is True
    assert r["persist_before_admit"] is True
    assert r["content_tamper_breaks"] is True
    assert r["reorder_breaks"] is True


def test_export_and_caveat() -> None:
    r = ledger_audit()
    assert r["export_shape"] is True
    assert r["export_counts"] is True
    assert r["export_head"] is True
    assert r["empty_valid_genesis"] is True
    # pinned caveat: tail truncation verifies clean without a head pin
    assert r["tail_truncation_invisible"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = ledger_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "ledger_audit_test.json")
    assert result["valid"], result.get("errors")
