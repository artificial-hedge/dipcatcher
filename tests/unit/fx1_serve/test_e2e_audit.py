"""Tests for fx1.serve.e2e_audit — real-socket lifecycle audit."""

from __future__ import annotations

import pytest

from fx1.serve.e2e_audit import e2e_audit, e2e_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "FX1_API_KEY",
        "FX1_BYOK_BASE_URL",
        "FX1_BYOK_API_KEY",
        "FX1_BYOK_MODEL",
        "MOONSHOT_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_all_probes_hold() -> None:
    results = e2e_audit()
    assert len(results) >= 14
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = e2e_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = e2e_audit_bench()
    b = e2e_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]
