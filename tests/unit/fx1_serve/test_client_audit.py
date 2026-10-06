"""Tests for fx1.serve.client_audit — the client error-boundary battery."""

from __future__ import annotations

from typing import Any

import pytest

from fx1.serve.client_audit import client_audit, client_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    """Run the expensive end-to-end battery once per module."""
    return client_audit()


def test_all_probes_hold(results: dict[str, Any]) -> None:
    assert len(results) >= 150
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies(results: dict[str, Any]) -> None:
    blob = client_audit_bench(results)
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic(results: dict[str, Any]) -> None:
    first = client_audit_bench(results)["receipt_sha256"]
    second = client_audit_bench(results)["receipt_sha256"]
    assert first == second


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
    blob = client_audit_bench(invalid_results)
    assert blob["claim"]["ok"] is False


def test_committed_receipt_still_verifies() -> None:
    import json
    from pathlib import Path

    path = Path("receipts/fx1_client_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []
