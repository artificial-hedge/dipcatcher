"""Tests for fx1.serve.webhook_delivery_audit — dispatcher internals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import webhook_delivery_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    """Run the scripted-server battery once per module — loopback
    http.server + patched clocks, deterministic."""
    return audit.webhook_delivery_audit()


def test_all_probes_hold(results: dict[str, Any]) -> None:
    assert len(results) == 100
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_dispatcher_seams(results: dict[str, Any]) -> None:
    for name in (
        "verify_rejects_reserialized_body",
        "signed_stale_still_refused",
        "literal_private_refused_0",
        "env_allows_loopback_literal",
        "env_restored_after_leg",
        "v4_mapped_private_refused",
        "http_connect_dials_validated_ip",
        "https_sni_is_url_host",
        "fail_fail_ok_delivers_third",
        "backoff_doubles",
        "definitive_4xx_no_retry",
        "fresh_ts_per_attempt",
        "fault_walks_to_next_address",
        "five_xx_short_circuits_walk",
        "conn_refused_retries_then_errors",
        "private_url_refused_zero_attempts",
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
    blob = audit.webhook_delivery_audit_bench(invalid_results)
    assert blob["claim"]["ok"] is False


def test_receipt_verifies(results: dict[str, Any]) -> None:
    blob = audit.webhook_delivery_audit_bench(results)
    assert blob["claim"]["ok"] is True
    assert verify_receipt_payload(blob)["valid"] is True


def test_receipt_deterministic(results: dict[str, Any]) -> None:
    a = audit.webhook_delivery_audit_bench(results)
    b = audit.webhook_delivery_audit_bench(results)
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_committed_receipt_verifies_without_rerunning() -> None:
    path = Path("receipts/fx1_webhook_delivery_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []


def test_private_env_never_leaks() -> None:
    """The SSRF opt-in must not survive the battery — every leg scopes
    and restores it."""
    import os

    assert os.environ.get("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS") in (None, "", "0")
