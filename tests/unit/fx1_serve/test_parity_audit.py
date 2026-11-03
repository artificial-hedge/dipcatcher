"""Tests for fx1.serve.parity_audit — SDK / API / HarnessClient parity."""

from __future__ import annotations

from typing import Any

import pytest

from fx1.serve.parity_audit import parity_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload
from tests.unit.fx1_serve.audit_session import run_with_clean_env

_AUDIT_ENV = (
    "FX1_API_KEY",
    "FX1_BYOK_BASE_URL",
    "FX1_BYOK_API_KEY",
    "FX1_BYOK_MODEL",
    "FX1_LOCAL_SERVE_URL",
    "FX1_LOCAL_SERVE_CMD",
    "FX1_LOCAL_MODEL",
    "FX1_LOCAL_API_KEY",
    "FX1_CHECKPOINT_DIR",
    "MOONSHOT_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _AUDIT_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def parity_bench() -> dict[str, Any]:
    return run_with_clean_env(parity_audit_bench, _AUDIT_ENV)


def test_all_probes_hold(parity_bench: dict[str, Any]) -> None:
    results = parity_bench["claim"]["results"]
    assert len(results) >= 40
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies(parity_bench: dict[str, Any]) -> None:
    assert parity_bench["claim"]["ok"] is True
    verdict = verify_receipt_payload(parity_bench)
    assert verdict["valid"] is True


def test_receipt_deterministic(parity_bench: dict[str, Any]) -> None:
    second = parity_audit_bench()
    assert parity_bench["receipt_sha256"] == second["receipt_sha256"]
