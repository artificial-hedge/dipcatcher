"""Tests for the apisurface_audit lane."""

from __future__ import annotations

import pytest

from fx1.serve.apisurface_audit import (
    apisurface_audit,
    apisurface_audit_bench,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "FX1_API_KEY",
        "FX1_ADMIN_KEY",
        "FX1_BYOK_BASE_URL",
        "FX1_BYOK_API_KEY",
        "FX1_BYOK_MODEL",
        "FX1_API_CORS_ORIGINS",
        "MOONSHOT_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_all_probes_hold() -> None:
    results = apisurface_audit()
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = apisurface_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = apisurface_audit_bench()
    b = apisurface_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_empty_audit_result_is_not_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fx1.serve.apisurface_audit.apisurface_audit", lambda: {})
    blob = apisurface_audit_bench()
    assert blob["claim"]["ok"] is False
    assert "no_probes" in blob["interpretation"]
