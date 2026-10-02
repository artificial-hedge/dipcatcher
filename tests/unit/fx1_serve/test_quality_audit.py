"""quality_audit lane: corpus quality-gate contract + caveats pinned."""

from __future__ import annotations

from fx1.data.quality_audit import quality_audit, quality_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_gates() -> None:
    r = quality_audit()
    assert r["exact_dedup"] is True
    assert r["contamination_flags"] is True
    assert r["nfkc_evasion_closed"] is True
    assert r["near_dup_windowed"] is True


def test_caveats_pinned() -> None:
    r = quality_audit()
    # inverted shingle index: a dup >500 kept items later is still caught
    assert r["far_apart_dup_caught"] is True
    # overlength drops are uncounted in the report
    assert r["overlong_silent_drop"] is True


def test_split() -> None:
    r = quality_audit()
    assert r["split_deterministic"] is True
    assert r["split_counts"] is True
    assert r["digest_binds_bytes"] is True
    assert r["bounds_enforced"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = quality_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "quality_audit_test.json")
    assert result["valid"], result.get("errors")
