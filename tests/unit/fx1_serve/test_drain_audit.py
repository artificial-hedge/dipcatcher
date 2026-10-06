"""Tests for fx1.serve.drain_audit — the /harness/drain lifecycle surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fx1.serve.drain_audit import drain_audit, drain_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
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
        "FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    """Run the expensive end-to-end battery once per module."""
    return drain_audit()


def test_all_probes_hold(results: dict[str, Any]) -> None:
    assert len(results) >= 150
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_emergency_key_revocation_stays_open(results: dict[str, Any]) -> None:
    assert results["revoke_key_open"] is True


def test_receipt_verifies(results: dict[str, Any]) -> None:
    blob = drain_audit_bench(results)
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic(results: dict[str, Any]) -> None:
    a = drain_audit_bench(results)
    b = drain_audit_bench(results)
    assert a["receipt_sha256"] == b["receipt_sha256"]


@pytest.mark.parametrize(
    "invalid_results",
    [
        {},
        {"probe": False},
        {"probe": 1},
        {"probe": None},
    ],
)
def test_receipt_fails_closed_on_invalid_results(invalid_results: dict[str, Any]) -> None:
    blob = drain_audit_bench(invalid_results)
    assert blob["claim"]["ok"] is False


def test_committed_receipt_still_verifies() -> None:
    path = Path("receipts/fx1_drain_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []
