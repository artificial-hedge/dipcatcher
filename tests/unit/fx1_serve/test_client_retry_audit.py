"""Tests for fx1.serve.client_retry_audit — HarnessClient retry mechanics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import client_retry_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    """Run the scripted-transport battery once per module — pure CPU,
    no sockets or wall-clock sleeps."""
    return audit.client_retry_audit()


def test_all_probes_hold(results: dict[str, Any]) -> None:
    assert len(results) == 66
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_retry_schedule_seams(results: dict[str, Any]) -> None:
    for name in (
        "backoff_doubles_pure_faults",
        "ra_overrides_grown_backoff",
        "ra_at_cap_retries",
        "ra_over_cap_breaks_no_sleep",
        "sleep_precedes_retry",
        "mapped_503_keeps_headers",
        "mapped_500_clears_headers",
        "mapped_503_exhaustion_skips_circuit",
        "idempotency_key_on_every_attempt",
    ):
        assert results[name] is True, f"seam {name} uncovered"


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
    blob = audit.client_retry_audit_bench(invalid_results)
    assert blob["claim"]["ok"] is False


def test_receipt_verifies(results: dict[str, Any]) -> None:
    blob = audit.client_retry_audit_bench(results)
    assert blob["claim"]["ok"] is True
    assert verify_receipt_payload(blob)["valid"] is True


def test_receipt_deterministic(results: dict[str, Any]) -> None:
    a = audit.client_retry_audit_bench(results)
    b = audit.client_retry_audit_bench(results)
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_committed_receipt_verifies_without_rerunning() -> None:
    path = Path("receipts/fx1_client_retry_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []


def test_scripted_transport_repeats_last_step() -> None:
    """The fault-forever leg: a lone exception step must repeat, not
    degrade into the 500 fallback — the retry probes lean on it."""
    from fx1.serve.client import HarnessTransportError

    fault = HarnessTransportError("dial failed")
    tr, calls, _ = audit._scripted(fault)
    cli = audit._mk(tr, max_retries=2, sleep=lambda s: None)
    exc = None
    try:
        cli.commands()
    except HarnessTransportError as e:
        exc = e
    assert exc is not None and "dial failed" in str(exc)
    assert len(calls) == 3
