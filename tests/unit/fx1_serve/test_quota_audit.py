"""Synthetic quota boundaries and the audit's evidence/isolation helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import quota_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def measured() -> dict[str, Any]:
    return audit.quota_audit()


def test_contract_probes_hold(measured: dict[str, Any]) -> None:
    assert len(measured) == 93
    assert all(value is True for value in measured.values()), measured


def test_durable_lifetime_and_process_local_window(measured: dict[str, Any]) -> None:
    for name in (
        "persisted_uses_after_restart",
        "persisted_tokens_after_restart",
        "persisted_remaining_budget_enforced",
        "persisted_spenddown_to_refusal",
        "persisted_refusal_is_quota",
        "store_replay_reconstructs_counters",
        "process_local_window_resets_after_restart",
        "restarted_window_admits_with_durable_spend",
        "store_replay_window_is_process_local",
    ):
        assert measured[name] is True, name


@pytest.mark.parametrize("results", [{}, {"probe": False}, {"probe": 1}, {"probe": None}])
def test_receipt_refuses_empty_or_nonliteral_success(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, Any]
) -> None:
    monkeypatch.setattr(audit, "quota_audit", lambda: results)
    receipt = audit.quota_audit_bench()
    assert receipt["claim"]["ok"] is False
    assert receipt["claim"]["results"] == results
    assert receipt["interpretation"].startswith("QUOTA AUDIT DEFECT:")
    assert verify_receipt_payload(receipt)["valid"] is True


def test_measured_receipt_verifies_without_rerunning(
    measured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "quota_audit", lambda: measured)
    receipt = audit.quota_audit_bench()
    assert receipt["claim"]["ok"] is True
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert verify_receipt_payload(receipt)["valid"] is True
    assert receipt == audit.quota_audit_bench()
    assert "TypeScript client runtime" in receipt["coverage"]["not_executed"]


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


def test_retry_after_parser_rejects_invalid_values() -> None:
    class Response:
        headers: dict[str, str] = {}

    response = Response()
    assert audit._retry_after(response) is None
    response.headers = {"retry-after": "nonsense"}
    assert audit._retry_after(response) is None
    response.headers = {"retry-after": "7"}
    assert audit._retry_after(response) == 7
