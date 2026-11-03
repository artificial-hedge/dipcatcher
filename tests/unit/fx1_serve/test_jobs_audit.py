"""Tests for fx1.serve.jobs_audit — /harness/jobs + /harness/runs surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fx1.serve.jobs_audit import jobs_audit_bench
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
    "FX1_API_STATE_DIR",
    "FX1_SDK_STATE_DIR",
    "FX1_CHECKPOINT_DIR",
    "MOONSHOT_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _AUDIT_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def jobs_bench() -> dict[str, Any]:
    return run_with_clean_env(jobs_audit_bench, _AUDIT_ENV)


def test_all_probes_hold(jobs_bench: dict[str, Any]) -> None:
    results = jobs_bench["claim"]["results"]
    assert len(results) >= 190
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies(jobs_bench: dict[str, Any]) -> None:
    assert jobs_bench["claim"]["ok"] is True
    verdict = verify_receipt_payload(jobs_bench)
    assert verdict["valid"] is True


def test_receipt_deterministic(jobs_bench: dict[str, Any]) -> None:
    second = jobs_audit_bench()
    assert jobs_bench["receipt_sha256"] == second["receipt_sha256"]


def test_committed_receipt_still_verifies() -> None:
    path = Path("receipts/fx1_jobs_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []
