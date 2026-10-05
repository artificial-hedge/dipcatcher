"""Tests for fx1.serve.usage_audit — the accounting-conservation battery."""

from __future__ import annotations

from fx1.serve.usage_audit import usage_audit, usage_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_all_probes_hold() -> None:
    results = usage_audit()
    assert len(results) >= 30
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = usage_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    first = usage_audit_bench()["receipt_sha256"]
    second = usage_audit_bench()["receipt_sha256"]
    assert first == second


def test_committed_receipt_still_verifies() -> None:
    import json
    from pathlib import Path

    path = Path("receipts/fx1_usage_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []
