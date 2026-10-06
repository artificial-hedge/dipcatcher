"""Tests for fx1.serve.ops_audit — the ops-surface probe battery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fx1.serve.ops_audit import ops_audit, ops_audit_bench
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
    return ops_audit()


def test_all_probes_hold(results: dict[str, Any]) -> None:
    assert len(results) >= 135
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_anthropic_dialect_scoped_off_ops(results: dict[str, Any]) -> None:
    """The fixed leak: an anthropic-version header must not stamp dialect
    headers onto ops answers."""
    assert results["anthropic_dialect_absent_on_metrics"] is True
    assert results["anthropic_dialect_absent_on_ready"] is True
    assert results["anthropic_dialect_intact_on_v1"] is True


def test_probe_verdicts_meter(results: dict[str, Any]) -> None:
    """The fixed metering gap: resolve-stage verdicts count under
    ``probe:<name>`` too."""
    assert results["probe_meters_own_series"] is True


def test_receipt_verifies(results: dict[str, Any]) -> None:
    blob = ops_audit_bench(results)
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic(results: dict[str, Any]) -> None:
    a = ops_audit_bench(results)
    b = ops_audit_bench(results)
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
    blob = ops_audit_bench(invalid_results)
    assert blob["claim"]["ok"] is False


def test_committed_receipt_still_verifies() -> None:
    path = Path("receipts/fx1_ops_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []
