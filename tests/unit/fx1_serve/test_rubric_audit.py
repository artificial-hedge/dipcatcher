"""Tests for fx1.rubric_audit."""

from __future__ import annotations

from fx1.rubric_audit import rubric_audit, rubric_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_all_probes_hold() -> None:
    results = rubric_audit()
    assert len(results) >= 30
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = rubric_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = rubric_audit_bench()
    b = rubric_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]
