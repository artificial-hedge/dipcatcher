"""Model-registry audit evidence and isolated resource lifecycle checks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import models_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def measured() -> dict[str, bool]:
    return audit.models_audit()


def test_contract_probes_hold(measured: dict[str, bool]) -> None:
    assert len(measured) == 157
    assert all(value is True for value in measured.values()), measured


def test_tombstone_and_durability_probes_hold(measured: dict[str, bool]) -> None:
    assert measured["ft_delete_tombstones_get"] is True
    assert measured["ft_delete_tombstones_list"] is True
    assert measured["ft_delete_tombstones_resolve"] is True
    assert measured["ft_delete_drops_checkpoints"] is True
    assert measured["inflight_completes_across_delete"] is True
    assert measured["inflight_new_request_404s"] is True
    assert measured["dur_registry_survives"] is True
    assert measured["dur_tombstone_stays_dead"] is True
    assert measured["dur_base_never_tombstoned"] is True


def test_registry_defect_regressions_stay_pinned(measured: dict[str, bool]) -> None:
    """The four defects this battery found stay fixed: honest ft card
    ``created``, no silent default link on unknown models, the ``ft:``
    alias on the response ``model``, and empty-model validation."""
    assert measured["ft_card_created_is_registration"] is True
    assert measured["param_unknown_chat_404"] is True
    assert measured["param_unknown_never_resolved"] is True
    assert measured["param_empty_chat_422"] is True
    assert measured["ft_response_model_is_ft_name"] is True
    assert measured["sdk_ft_response_model"] is True
    assert measured["ckpt_dir_only_local_fx1"] is True


@pytest.mark.parametrize("results", [{}, {"probe": False}, {"probe": 1}, {"probe": None}])
def test_empty_or_nonliteral_audit_never_succeeds(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, Any]
) -> None:
    monkeypatch.setattr(audit, "models_audit", lambda: results)
    receipt = audit.models_audit_bench()
    assert receipt["claim"]["ok"] is False
    assert receipt["claim"]["results"] == results
    assert verify_receipt_payload(receipt)["valid"] is True


def test_measured_receipt_verifies_without_rerunning(
    measured: dict[str, bool], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "models_audit", lambda: measured)
    receipt = audit.models_audit_bench()
    assert receipt["claim"]["ok"] is True
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert verify_receipt_payload(receipt)["valid"] is True
    assert receipt == audit.models_audit_bench()


def test_client_resources_and_ambient_state_survive_probe_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ambient = tmp_path / "operator"
    ambient.mkdir()
    sentinel = ambient / "ft_jobs.jsonl"
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
        client, _ = audit._client(audit._spy_resolver())
        assert client.get("/v1/models").status_code == 200
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
