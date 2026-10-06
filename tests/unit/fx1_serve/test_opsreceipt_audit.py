"""Tests for the opsreceipt_audit lane."""

from __future__ import annotations

import pytest

from fx1.serve.opsreceipt_audit import opsreceipt_audit, opsreceipt_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "FX1_API_KEY",
        "FX1_ADMIN_KEY",
        "MOONSHOT_API_KEY",
        "FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_all_probes_hold() -> None:
    results = opsreceipt_audit()
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = opsreceipt_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = opsreceipt_audit_bench()
    b = opsreceipt_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_empty_audit_result_is_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fx1.serve.opsreceipt_audit.opsreceipt_audit", lambda: {})
    blob = opsreceipt_audit_bench()
    assert blob["claim"]["ok"] is False
    assert "no_probes" in blob["interpretation"]
