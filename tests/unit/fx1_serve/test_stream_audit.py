"""Tests for fx1.serve.stream_audit."""

from __future__ import annotations

from typing import Any

import pytest

from fx1.serve.stream_audit import _ENV_KEYS, stream_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload
from tests.unit.fx1_serve.audit_session import run_with_clean_env


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_KEYS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def stream_bench() -> dict[str, Any]:
    return run_with_clean_env(stream_audit_bench, _ENV_KEYS)


def test_all_probes_hold(stream_bench: dict[str, Any]) -> None:
    results = stream_bench["claim"]["results"]
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies(stream_bench: dict[str, Any]) -> None:
    assert stream_bench["claim"]["ok"] is True
    verdict = verify_receipt_payload(stream_bench)
    assert verdict["valid"] is True


def test_receipt_deterministic(stream_bench: dict[str, Any]) -> None:
    second = stream_audit_bench()
    assert stream_bench["receipt_sha256"] == second["receipt_sha256"]


def test_empty_audit_result_is_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fx1.serve.stream_audit.stream_audit", lambda: {})
    blob = stream_audit_bench()
    assert blob["claim"]["ok"] is False
    assert "no_probes" in blob["interpretation"]
