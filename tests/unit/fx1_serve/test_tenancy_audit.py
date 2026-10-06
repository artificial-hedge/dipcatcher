"""Cross-key tenancy audit evidence and isolation contract checks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import tenancy_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def measured() -> dict[str, bool]:
    return audit.tenancy_audit()


def test_contract_probes_hold(measured: dict[str, bool]) -> None:
    assert len(measured) == 141
    assert all(value is True for value in measured.values()), measured


def test_shared_workspace_cells_stay_pinned(measured: dict[str, bool]) -> None:
    """The declared contract is scope-based workspace sharing: any key
    holding the verb's scope reads, mutates, and deletes any resource."""
    assert measured["responses_b_reads_a"] is True
    assert measured["responses_b_deletes_a"] is True
    assert measured["conversations_b_mutates_a"] is True
    assert measured["responses_b_binds_a_conversation"] is True
    assert measured["files_b_reads_a_content"] is True
    assert measured["uploads_b_completes_a"] is True
    assert measured["evals_b_runs_a_spec"] is True
    assert measured["registry_b_tombstones_a_model"] is True
    assert measured["vector_stores_b_deletes_a"] is True
    assert measured["usage_global_readable_by_any_scoped_key"] is True


def test_fixed_defect_regressions_stay_pinned(measured: dict[str, bool]) -> None:
    """The background-attribution defect this lane fixed stays fixed:
    background turns keep their principal in the ledger and on the
    meter, including past a mid-flight revocation."""
    assert measured["background_completion_carries_credential"] is True
    assert measured["background_spend_metered_on_owner"] is True
    assert measured["inflight_completion_keeps_actor_fingerprint"] is True
    assert measured["inflight_spend_metered_on_tombstone"] is True


def test_isolation_boundary_probes_hold(measured: dict[str, bool]) -> None:
    """Where isolation is claimed, it holds: authentication, scopes,
    per-key budgets, revocation, and key material never escaping."""
    assert measured["mixed_auth_refused_on_v1_400"] is True
    assert measured["mixed_auth_refused_off_v1_400"] is True
    assert measured["forged_x_api_key_with_bearer_refused_400"] is True
    assert measured["write_does_not_imply_read"] is True
    assert measured["cross_key_action_bills_actor"] is True
    assert measured["rpm_window_isolated_from_sibling"] is True
    assert measured["same_name_sibling_unaffected"] is True
    assert measured["roster_never_returns_key_material"] is True
    assert measured["peer_views_never_carry_key_material"] is True
    assert measured["drain_latches_globally_for_all_keys"] is True
    assert measured["tombstones_persist_across_restart"] is True
    assert measured["idem_replay_isolated_per_credential"] is True
    assert measured["idem_conflict_isolated_per_credential"] is True


@pytest.mark.parametrize("results", [{}, {"probe": False}, {"probe": 1}, {"probe": None}])
def test_empty_or_nonliteral_audit_never_succeeds(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, Any]
) -> None:
    monkeypatch.setattr(audit, "tenancy_audit", lambda: results)
    receipt = audit.tenancy_audit_bench()
    assert receipt["claim"]["ok"] is False
    assert receipt["claim"]["results"] == results
    assert verify_receipt_payload(receipt)["valid"] is True


def test_measured_receipt_verifies_without_rerunning(
    measured: dict[str, bool], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "tenancy_audit", lambda: measured)
    receipt = audit.tenancy_audit_bench()
    assert receipt["claim"]["ok"] == all(v is True for v in measured.values())
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert verify_receipt_payload(receipt)["valid"] is True
    assert receipt == audit.tenancy_audit_bench()


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
        client, _ = audit._client()
        assert client.get("/v1/models", headers=audit._h(audit._ROOT)).status_code == 200
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
