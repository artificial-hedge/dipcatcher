"""Tests for fx1.serve.retrieval_audit — the retrieval sub-resource battery."""

from __future__ import annotations

from typing import Any

import pytest

from fx1.serve import openai_compat, vectorstores
from fx1.serve import retrieval_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload

_NAMED_PINS = frozenset(
    {
        # empty vs missing
        "conv_items_empty_page",
        "output_items_inflight_empty_not_404",
        "output_items_nonterminal_empty_page",
        "ft_events_envelope_from_submit",
        "ft_checkpoints_empty_pre_lifecycle",
        "vs_files_empty_page",
        "job_logs_route_absent_404",
        "unlisted_surfaces_enveloped_404",
        # ordering
        "conv_items_insertion_order_asc",
        "conv_items_order_desc_flips",
        "ft_events_oldest_first_chronological",
        "evals_list_newest_first",
        "batches_list_newest_first",
        "vs_files_attach_order_asc",
        "vs_batch_files_request_order_frozen",
        # paging
        "conv_items_after_walk_covers_all",
        "conv_items_before_previous_page",
        "conv_items_before_walk_covers_all",
        "conv_items_limit_slice",
        "conv_items_cursor_fields_consistent",
        "conv_items_deleted_cursor_400",
        "conv_items_cursor_reuse_mutated_bounded",
        "ft_events_after_walk_covers",
        "ft_events_small_limit_walks_all",
        "output_items_cursor_paginates",
        "files_list_after_walk_covers",
        "vs_files_before_previous_page",
        "batches_list_after_exclusive",
        "models_anthropic_cursor_walk",
        # envelope
        "models_openai_envelope",
        "models_anthropic_grammar",
        "jobs_empty_inventory_shape",
        "evals_page_envelope_shape",
        "anthropic_missing_404_enveloped",
        "openai_missing_404s_enveloped",
        "harness_missing_404s_enveloped",
        # cursor dialect split — pinned, not unified
        "conv_items_cursor_400s_enveloped",
        "chat_messages_unknown_cursor_400",
        "files_list_unknown_cursor_400",
        "vs_list_unknown_cursor_400",
        "runs_list_unknown_cursor_400",
        "ft_events_unknown_cursor_empty_honest",
        "ft_jobs_unknown_cursor_empty",
        "ft_checkpoints_unknown_cursor_empty_honest",
        "abatches_unknown_cursor_empty",
        "models_anthropic_unknown_cursor_empty",
        # deletion boundary
        "conv_delete_tombstones_items",
        "chat_delete_tombstones_messages",
        "response_delete_tombstones_input_items",
        "spec_delete_tombstones_runs",
        "run_delete_tombstones_output_items",
        "vs_delete_tombstones_subresources",
        "abatch_delete_tombstones_results",
        "vs_detach_404_orphan_survives",
        "vs_detach_drops_from_listing",
        "ft_deleted_model_drops_checkpoint",
        # stream surfaces
        "results_jsonl_eof_terminated",
        "results_stream_replay_identical",
        "results_refused_until_ended_400",
        "job_events_sse_frames_end_terminal",
        "job_events_frames_complete_json",
        "job_events_unknown_404_before_stream",
        # monotonic lifecycle
        "ft_events_grow_monotonic",
        "output_items_populated_post_completion",
        "failed_tasks_output_items_honest_rows",
        "cancel_running_run_conflict_409",
        "ft_pause_marks_nonterminal",
        "ft_cancelled_job_events_honest_tail",
        "ft_cancelled_checkpoints_honest_empty",
        "ft_checkpoints_empty_running_then_registered",
        "ft_checkpoint_shape_first_last",
        "response_replay_monotonic_terminal",
        "response_replay_starting_after_resumes",
        # bounded sub-resources
        "ft_events_capped_at_declared_bound",
        "ft_events_retention_contiguous_suffix",
        "ft_events_walk_pages_honest",
        "ft_events_pages_never_exceed_limit",
        "ft_events_limit_bound_refused",
        # idempotent reads
        "conv_items_repeat_gets_identical",
        "ft_events_repeat_gets_identical",
        "output_items_repeat_gets_identical",
        "files_list_repeat_gets_identical",
        # scope
        "unauthenticated_read_401",
        "write_scoped_key_read_refused_403",
        "read_scoped_key_reads_all",
        "read_scoped_key_write_refused_403",
        "cross_key_visibility_shared_workspace",
        # concurrent read-during-write
        "conv_items_concurrent_counts_monotonic",
        "conv_items_concurrent_no_torn_reads",
        "ft_events_concurrent_consistent_snapshots",
        # misc list surfaces
        "chat_list_oldest_first_envelope",
        "chat_list_model_filter",
        "chat_list_metadata_filter",
        "jobs_list_newest_first_total",
        "jobs_list_offset_pages",
        "jobs_list_status_filter",
        "files_list_purpose_filter",
        "completions_list_newest_first_window",
        "runs_list_scoped_envelope",
        "input_items_populated_page",
        "input_items_desc_flips",
        "file_content_bytes_roundtrip",
        "vs_file_get_and_content",
        "vs_file_batch_terminal_create",
        "vs_batch_files_missing_404",
        "vs_batch_files_filter_honest",
        "vs_list_default_desc",
        "vs_list_order_asc_flips",
        "vs_files_desc_flips",
        "vs_files_filter_words",
        "evals_list_refusals_enveloped",
        "output_items_refusals_enveloped",
        "output_items_rows_index_ids",
        "runs_list_cursor_dual_grammar",
        "abatches_newest_first",
        "abatch_empty_envelope",
        "files_list_newest_first",
        "files_list_order_asc_flips",
        "ft_jobs_newest_first",
        "ft_jobs_after_exclusive",
        "evals_list_newest_first_envelope",
        "evals_list_after_exclusive",
        "chat_messages_envelope_and_ids",
        "chat_messages_desc_limit",
        "run_status_maps_wire_grammar",
        "batches_list_unknown_cursor_400",
    }
)


@pytest.fixture(scope="module")
def measured() -> dict[str, Any]:
    return audit.retrieval_audit()


def test_contract_probes_hold(measured: dict[str, Any]) -> None:
    assert len(measured) == 136
    assert all(value is True for value in measured.values()), measured


def test_named_pins_cover_every_retrieval_surface(measured: dict[str, Any]) -> None:
    assert measured.keys() >= _NAMED_PINS
    for name in sorted(_NAMED_PINS):
        assert measured[name] is True, name


@pytest.mark.parametrize("results", [{}, {"probe": False}, {"probe": 1}, {"probe": None}])
def test_receipt_refuses_empty_or_nonliteral_success(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, Any]
) -> None:
    monkeypatch.setattr(audit, "retrieval_audit", lambda: results)
    receipt = audit.retrieval_audit_bench()
    assert receipt["claim"]["ok"] is False
    assert receipt["claim"]["results"] == results
    assert receipt["interpretation"].startswith("RETRIEVAL AUDIT DEFECT:")
    assert verify_receipt_payload(receipt)["valid"] is True


def test_receipt_refuses_incomplete_all_true_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "retrieval_audit", lambda: {"probe": True})
    assert audit.retrieval_audit_bench()["claim"]["ok"] is False


@pytest.mark.parametrize("limit", [0, -1, 101])
def test_before_helpers_reject_invalid_limit(limit: int) -> None:
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    with pytest.raises(openai_compat.OpenAICompatError, match="limit must be 1..100"):
        openai_compat.paged_item_list(rows, limit=limit, before="c")
    with pytest.raises(vectorstores.VectorStoreError, match="limit must be 1..100"):
        vectorstores._page(rows, limit=limit, order="asc", after=None, before="c")


def test_measured_receipt_verifies_without_rerunning(
    measured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "retrieval_audit", lambda: measured)
    receipt = audit.retrieval_audit_bench()
    assert receipt["claim"]["ok"] is True
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert verify_receipt_payload(receipt)["valid"] is True
    assert receipt == audit.retrieval_audit_bench()


def test_committed_receipt_matches_fresh_measurement(
    measured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from pathlib import Path

    path = Path("receipts/fx1_retrieval_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []
    monkeypatch.setattr(audit, "retrieval_audit", lambda: measured)
    fresh = audit.retrieval_audit_bench()
    for field in (
        "kind",
        "schema",
        "data_label",
        "research_only",
        "live_pnl_claim",
        "claim",
        "coverage",
        "interpretation",
    ):
        assert payload[field] == fresh[field]
