"""Store-boundary audit evidence and persistence contract checks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import store_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def measured() -> dict[str, bool]:
    return audit.store_audit()


def test_contract_probes_hold(measured: dict[str, bool]) -> None:
    assert len(measured) == 112
    assert all(value is True for value in measured.values()), measured


def test_retrieval_boundary_probes_stay_pinned(measured: dict[str, bool]) -> None:
    """store=false mints ids that never resolve: every read, list,
    subresource, update and delete on the id answers the enveloped 404,
    while the stored controls retrieve."""
    assert measured["chat_store_false_get_404_enveloped"] is True
    assert measured["chat_store_false_messages_404"] is True
    assert measured["chat_store_false_list_excludes"] is True
    assert measured["chat_store_false_update_404"] is True
    assert measured["chat_store_false_delete_404_enveloped"] is True
    assert measured["chat_store_true_get_200"] is True
    assert measured["chat_store_default_pins"] is True
    assert measured["responses_store_false_get_404"] is True
    assert measured["responses_store_false_stream_replay_404"] is True
    assert measured["responses_store_false_input_items_404"] is True
    assert measured["responses_store_true_get_200"] is True
    assert measured["responses_store_true_stream_replay_streams"] is True


def test_chaining_and_container_probes_stay_pinned(measured: dict[str, bool]) -> None:
    """previous_response_id needs a stored parent; conversation keeps the
    turn under store=false because the conv is its own store; background
    refuses the combination outright."""
    assert measured["responses_prev_off_unstored_parent_400"] is True
    assert measured["responses_prev_off_unstored_parent_enveloped"] is True
    assert measured["responses_prev_off_stored_parent_200"] is True
    assert measured["responses_chained_child_carries_history"] is True
    assert measured["responses_conv_store_false_turn_appends"] is True
    assert measured["responses_conv_store_false_response_uncached"] is True
    assert measured["bg_store_false_400_requires_store"] is True
    assert measured["bg_stream_store_false_runs_as_stream"] is True
    assert measured["bg_store_true_queued"] is True


def test_inert_and_idempotency_probes_stay_pinned(measured: dict[str, bool]) -> None:
    """The no-knob surfaces pin the flag inert at translation, and the
    idempotency ledger replays store=false calls without repinning them
    or re-spending the backend."""
    assert measured["messages_derived_chat_id_unretrievable"] is True
    assert measured["messages_store_true_extra_inert"] is True
    assert measured["legacy_derived_chat_id_unretrievable"] is True
    assert measured["legacy_no_retrieval_twin_404_enveloped"] is True
    assert measured["idem_chat_store_false_replays"] is True
    assert measured["idem_chat_store_false_no_reexecution"] is True
    assert measured["idem_chat_store_false_still_unpinned"] is True
    assert measured["idem_responses_store_false_replays"] is True
    assert measured["idem_chat_store_false_stream_replays"] is True
    assert measured["idem_store_false_resume_replays_tail"] is True
    assert measured["idem_legacy_replay_is_retrieval_path"] is True
    assert measured["idem_shared_ledger_cross_surface_conflict_409"] is True


def test_durability_asymmetry_probes_stay_pinned(measured: dict[str, bool]) -> None:
    """The restart map is pinned as documented: the retrieval index and
    completion log are process-local, while conversations, idem records,
    meters and eval/batch stores are journaled and survive."""
    assert measured["restart_index_drops_stored_chat"] is True
    assert measured["restart_index_drops_stored_response"] is True
    assert measured["restart_store_false_still_404"] is True
    assert measured["restart_conversations_survive"] is True
    assert measured["restart_conv_items_include_store_false_turn"] is True
    assert measured["restart_idem_replays_store_false"] is True
    assert measured["restart_eval_run_survives"] is True
    assert measured["restart_key_meters_survive"] is True
    assert measured["restart_completion_log_process_local"] is True
    assert measured["batch_chat_line_store_false_get_404"] is True
    assert measured["batch_chat_line_stored_get_200"] is True
    assert measured["batch_responses_line_store_false_get_404"] is True
    assert measured["metering_store_false_bills_uses"] is True
    assert measured["metering_store_false_bills_tokens"] is True
    assert measured["metering_store_false_row_key_attributed"] is True
    assert measured["metering_parity_store_true"] is True


@pytest.mark.parametrize("results", [{}, {"probe": False}, {"probe": 1}, {"probe": None}])
def test_empty_or_nonliteral_audit_never_succeeds(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, Any]
) -> None:
    monkeypatch.setattr(audit, "store_audit", lambda: results)
    receipt = audit.store_audit_bench()
    assert receipt["claim"]["ok"] is False
    assert receipt["claim"]["results"] == results
    assert verify_receipt_payload(receipt)["valid"] is True


def test_measured_receipt_verifies_without_rerunning(
    measured: dict[str, bool], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "store_audit", lambda: measured)
    receipt = audit.store_audit_bench()
    assert receipt["claim"]["ok"] == all(v is True for v in measured.values())
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert verify_receipt_payload(receipt)["valid"] is True
    assert receipt == audit.store_audit_bench()


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
