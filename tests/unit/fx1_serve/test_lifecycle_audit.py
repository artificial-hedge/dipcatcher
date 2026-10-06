"""Tests for fx1.serve.lifecycle_audit — the process-lifecycle battery."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from fx1.serve.lifecycle_audit import lifecycle_audit, lifecycle_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.startswith(("FX1_", "MOONSHOT_")):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    """Run the expensive end-to-end battery once per module."""
    return lifecycle_audit()


def test_all_probes_hold(results: dict[str, Any]) -> None:
    assert len(results) >= 80
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_sigterm_exit_is_by_signal(results: dict[str, Any]) -> None:
    assert results["sigterm_exits_by_signal"] is True


def test_shutdown_grace_is_bounded(results: dict[str, Any]) -> None:
    assert results["grace_exit_on_bound_not_signal"] is True


def test_double_boot_fails_closed(results: dict[str, Any]) -> None:
    assert results["double_boot_refused_nonzero"] is True
    assert results["first_process_unaffected"] is True


def test_receipt_verifies(results: dict[str, Any]) -> None:
    blob = lifecycle_audit_bench(results)
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic(results: dict[str, Any]) -> None:
    a = lifecycle_audit_bench(results)
    b = lifecycle_audit_bench(results)
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
    blob = lifecycle_audit_bench(invalid_results)
    assert blob["claim"]["ok"] is False


def test_committed_receipt_still_verifies() -> None:
    path = Path("receipts/fx1_lifecycle_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []
