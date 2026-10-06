"""Synthetic introspection-truthfulness probes and the audit's helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import quota2_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def measured() -> dict[str, Any]:
    return audit.quota2_audit()


def test_contract_probes_hold(measured: dict[str, Any]) -> None:
    assert len(measured) == 107
    assert all(value is True for value in measured.values()), measured


def test_reset_surface_legs(measured: dict[str, Any]) -> None:
    for name in (
        "card_reports_window_reset_s",
        "reset_anchored_near_window_len",
        "fresh_window_reset_zero",
        "unwindowed_no_reset_surface",
        "card_reset_is_relative_only",
        "header_reset_agrees_with_card",
        "refusal_reset_consistent_with_retry",
        "refusal_card_window_full",
        "anthropic_reset_absolute_instant",
        "quota_budget_reports_no_reset",
    ):
        assert measured[name] is True, name


def test_window_policy_and_recovery(measured: dict[str, Any]) -> None:
    for name in (
        "window_fixed_anchor_drops_whole",
        "window_expiry_readmits",
        "window_recovery_card_fresh",
        "patch_rpm_relief_admits",
        "patch_rpm_card_window_null",
    ):
        assert measured[name] is True, name


def test_metering_and_persistence_truth(measured: dict[str, Any]) -> None:
    for name in (
        "rate_refusal_bills_nothing",
        "rate_refusal_no_backend_call",
        "postauth_422_bills_use_not_tokens",
        "retry_sequence_bills_admitted_only",
        "tokens_equal_provider_sum",
        "persist_uses_across_restart",
        "persist_last_used_at_exact",
        "persist_window_declared_process_local",
        "persist_served_process_local",
        "persist_next_use_counts_durable",
    ):
        assert measured[name] is True, name


def test_distinction_independence_and_clock(measured: dict[str, Any]) -> None:
    for name in (
        "quota_refusal_no_rate_headers",
        "quota_refusal_distinct_from_rpm",
        "self_depleted_is_quota_refusal",
        "peer_quota_isolated",
        "peer_window_isolated",
        "peer_tokens_isolated",
        "created_at_consistent_across_legs",
        "client_headers_cannot_stamp",
        "admin_read_never_bills_target",
        "uses_counts_reads_not_served",
    ):
        assert measured[name] is True, name


@pytest.mark.parametrize("results", [{}, {"probe": False}, {"probe": 1}, {"probe": None}])
def test_receipt_refuses_empty_or_nonliteral_success(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, Any]
) -> None:
    monkeypatch.setattr(audit, "quota2_audit", lambda: results)
    receipt = audit.quota2_audit_bench()
    assert receipt["claim"]["ok"] is False
    assert receipt["claim"]["results"] == results
    assert receipt["interpretation"].startswith("QUOTA2 AUDIT DEFECT:")
    assert verify_receipt_payload(receipt)["valid"] is True


def test_measured_receipt_verifies_without_rerunning(
    measured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "quota2_audit", lambda: measured)
    receipt = audit.quota2_audit_bench()
    assert receipt["claim"]["ok"] is True
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert verify_receipt_payload(receipt)["valid"] is True
    assert receipt == audit.quota2_audit_bench()
    assert "TypeScript client runtime" in receipt["coverage"]["not_executed"]


def test_committed_receipt_still_verifies() -> None:
    import json

    path = Path("receipts/fx1_quota2_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["schema"] == "quota2_audit.v1"
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []


def test_audit_context_restores_environment_and_removes_temp_state_on_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = tmp_path / "operator_state"
    state.mkdir()
    marker = state / "keys.jsonl"
    marker.write_bytes(b"operator data must remain untouched\n")
    before = marker.read_bytes()
    env = {
        "FX1_API_STATE_DIR": str(state),
        "FX1_API_RECEIPTS_DIR": str(state),
        "FX1_FT_DIR": str(state),
        "FX1_API_CORS_ORIGINS": "*",
        "FX1_API_JOB_MAX": "not-a-number",
        "MOONSHOT_API_KEY": "synthetic ambient sentinel",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    temporary: Path | None = None
    with pytest.raises(RuntimeError, match="deliberate failure"), audit._audit_context():
        assert all(name not in os.environ for name in env)
        temporary = audit._temporary_directory()
        (temporary / "test.txt").write_text("synthetic")
        os.environ["FX1_TEMPORARY_AUDIT_VALUE"] = "temporary"
        raise RuntimeError("deliberate failure")
    assert temporary is not None and not temporary.exists()
    assert all(os.environ[name] == value for name, value in env.items())
    assert "FX1_TEMPORARY_AUDIT_VALUE" not in os.environ
    assert marker.read_bytes() == before


def test_client_uses_private_state_and_closes_executor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    operator = tmp_path / "operator"
    operator.mkdir()
    marker = operator / "keys.jsonl"
    marker.write_text("operator sentinel\n")
    monkeypatch.setenv("FX1_API_STATE_DIR", str(operator))
    monkeypatch.setenv("FX1_FT_DIR", str(operator))
    monkeypatch.setenv("FX1_API_RECEIPTS_DIR", str(operator))
    monkeypatch.setenv("FX1_API_STORE_MAX", "invalid ambient bound")
    with audit._audit_context():
        client, _ = audit._client(
            {"byok": lambda: audit._MeterBackend("synthetic", dict(audit._U6))},
            api_key=audit._ROOT,
        )
        raw, _ = audit._mint(client, {audit._H_KEY: audit._ROOT}, max_requests=1)
        assert audit._complete(client, {audit._H_KEY: raw}).status_code == 200
        executor = client.app.state.jobs_executor
        assert executor.submit(lambda: 3).result(timeout=1) == 3
    assert client.is_closed
    with pytest.raises(RuntimeError, match="shutdown"):
        executor.submit(lambda: None)
    assert list(operator.iterdir()) == [marker]
    assert marker.read_text() == "operator sentinel\n"


def test_retry_after_and_rfc3339_parsers_reject_invalid_values() -> None:
    class Response:
        headers: dict[str, str] = {}

    response = Response()
    assert audit._retry_after(response) is None
    response.headers = {"retry-after": "nonsense"}
    assert audit._retry_after(response) is None
    response.headers = {"retry-after": "7"}
    assert audit._retry_after(response) == 7

    assert audit._rfc3339_instant(None) is None
    assert audit._rfc3339_instant("not-a-date") is None
    instant = audit._rfc3339_instant("2026-10-06T01:00:00Z")
    assert isinstance(instant, float)
