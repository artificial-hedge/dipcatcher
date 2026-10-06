"""Tests for the capabilities_audit lane."""

from __future__ import annotations

from fx1.capabilities_audit import capabilities_audit, capabilities_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_all_probes_hold() -> None:
    results = capabilities_audit()
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = capabilities_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = capabilities_audit_bench()
    b = capabilities_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_empty_audit_result_is_not_success(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("fx1.capabilities_audit.capabilities_audit", lambda: {})
    blob = capabilities_audit_bench()
    assert blob["claim"]["ok"] is False
    assert "no_probes" in blob["interpretation"]
