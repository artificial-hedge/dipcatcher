"""Tests for fx1.serve.e2e_audit — real-socket lifecycle audit."""

from __future__ import annotations

from typing import Any

import pytest

from fx1.serve.e2e_audit import e2e_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload
from tests.unit.fx1_serve.audit_session import run_with_clean_env

_AUDIT_ENV = (
    "FX1_API_KEY",
    "FX1_BYOK_BASE_URL",
    "FX1_BYOK_API_KEY",
    "FX1_BYOK_MODEL",
    "MOONSHOT_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _AUDIT_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def e2e_bench() -> dict[str, Any]:
    return run_with_clean_env(e2e_audit_bench, _AUDIT_ENV)


def test_all_probes_hold(e2e_bench: dict[str, Any]) -> None:
    results = e2e_bench["claim"]["results"]
    assert len(results) >= 14
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies(e2e_bench: dict[str, Any]) -> None:
    assert e2e_bench["claim"]["ok"] is True
    verdict = verify_receipt_payload(e2e_bench)
    assert verdict["valid"] is True


def test_receipt_deterministic(e2e_bench: dict[str, Any]) -> None:
    second = e2e_audit_bench()
    assert e2e_bench["receipt_sha256"] == second["receipt_sha256"]
