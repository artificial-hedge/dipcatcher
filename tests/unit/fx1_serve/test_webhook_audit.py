"""Tests for fx1.serve.webhook_audit — the signed-delivery contract lane."""

from __future__ import annotations

from typing import Any

import pytest

from fx1.serve.webhook_audit import webhook_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload
from tests.unit.fx1_serve.audit_session import run_with_clean_env

_AUDIT_ENV = ("FX1_API_KEY", "MOONSHOT_API_KEY", "FX1_CHECKPOINT_DIR")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _AUDIT_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def webhook_bench() -> dict[str, Any]:
    return run_with_clean_env(webhook_audit_bench, _AUDIT_ENV)


def test_all_probes_hold(webhook_bench: dict[str, Any]) -> None:
    results = webhook_bench["claim"]["results"]
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies(webhook_bench: dict[str, Any]) -> None:
    assert webhook_bench["claim"]["ok"] is True
    verdict = verify_receipt_payload(webhook_bench)
    assert verdict["valid"] is True


def test_receipt_deterministic(webhook_bench: dict[str, Any]) -> None:
    second = webhook_audit_bench()
    assert webhook_bench["receipt_sha256"] == second["receipt_sha256"]
