"""Tests for fx1.serve.webhookdel_audit — the delivery-dispatcher contract lane."""

from __future__ import annotations

import pytest

from fx1.serve.webhookdel_audit import webhookdel_audit, webhookdel_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FX1_API_KEY", "MOONSHOT_API_KEY", "FX1_CHECKPOINT_DIR"):
        monkeypatch.delenv(name, raising=False)


def test_all_probes_hold() -> None:
    results = webhookdel_audit()
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = webhookdel_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = webhookdel_audit_bench()
    b = webhookdel_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]
