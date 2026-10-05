"""Conversation audit evidence and isolated resource lifecycle checks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import conv_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def measured() -> dict[str, bool]:
    return audit.conv_audit()


def test_contract_probes_hold(measured: dict[str, bool]) -> None:
    assert len(measured) == 141
    assert all(value is True for value in measured.values()), measured


def test_recovery_refuses_corruption(measured: dict[str, bool]) -> None:
    assert measured["dur_torn_tail_fails_closed"] is True
    assert measured["dur_torn_tail_preserves_evidence"] is True
    assert measured["dur_conv_survives"] is True
    assert measured["dur_tombstone_survives"] is True


@pytest.mark.parametrize("results", [{}, {"probe": False}, {"probe": 1}, {"probe": None}])
def test_empty_or_nonliteral_audit_never_succeeds(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, Any]
) -> None:
    monkeypatch.setattr(audit, "conv_audit", lambda: results)
    receipt = audit.conv_audit_bench()
    assert receipt["claim"]["ok"] is False
    assert receipt["claim"]["results"] == results
    assert verify_receipt_payload(receipt)["valid"] is True


def test_measured_receipt_verifies_without_rerunning(
    measured: dict[str, bool], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "conv_audit", lambda: measured)
    receipt = audit.conv_audit_bench()
    assert receipt["claim"]["ok"] is True
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert verify_receipt_payload(receipt)["valid"] is True
    assert receipt == audit.conv_audit_bench()


def test_client_resources_and_ambient_state_survive_probe_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ambient = tmp_path / "operator"
    ambient.mkdir()
    sentinel = ambient / "conversations.jsonl"
    sentinel.write_text("operator data must remain unchanged\n")
    config = {
        "FX1_API_STATE_DIR": str(ambient),
        "FX1_API_RECEIPTS_DIR": str(ambient),
        "FX1_FT_DIR": str(ambient),
        "FX1_SDK_STATE_DIR": str(ambient),
        "FX1_API_JOB_MAX": "invalid ambient value",
        "MOONSHOT_API_KEY": "synthetic ambient sentinel",
    }
    for name, value in config.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="deliberate failure"), audit._audit_context():
        assert all(name not in os.environ for name in config)
        temporary = audit._temporary_directory()
        client, _ = audit._client({audit._MODEL: audit._StubBackend})
        audit._conv_create(client)
        executor = client.app.state.jobs_executor
        assert executor.submit(lambda: 2).result(timeout=1) == 2
        raise RuntimeError("deliberate failure")
    assert client.is_closed
    assert not temporary.exists()
    with pytest.raises(RuntimeError, match="shutdown"):
        executor.submit(lambda: None)
    assert all(os.environ[name] == value for name, value in config.items())
    assert list(ambient.iterdir()) == [sentinel]
    assert sentinel.read_text() == "operator data must remain unchanged\n"
