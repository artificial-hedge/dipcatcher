"""Tests for fx1.serve.schema_audit — the request-model validation battery."""

from __future__ import annotations

from typing import Any

import pytest

from fx1.serve import schema_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload

_NAMED_PINS = frozenset(
    {
        # Type strictness — wrong types refuse, parseable scalars coerce
        # honestly (the pinned lax policy, not a laundered "strict" one).
        "chat_temp_word",
        "chat_temp_list",
        "chat_max_tokens_float",
        "chat_n_string_coerced",
        "chat_stream_word_coerced_sse",
        "chat_temp_bool_coerced",
        "chat_max_tokens_bool_coerced",
        # Bounds — every declared edge refuses past and accepts at it.
        "chat_temp_below_zero",
        "chat_temp_above_two",
        "chat_temp_edge_two_ok",
        "chat_max_tokens_neg",
        "chat_max_tokens_zero",
        "chat_max_tokens_overflow",
        "chat_n_zero",
        "chat_n_above_cap",
        "chat_n_cap_ok",
        "chat_top_p_above",
        "chat_freq_penalty_high",
        "chat_pres_penalty_low",
        # Enum/literal — out-of-domain values refuse; role strictness is
        # honestly surface-dependent.
        "chat_rf_bogus",
        "chat_tool_choice_type_bogus",
        "chat_reasoning_ultra",
        "chat_role_sysadmin_accepted",
        "resp_role_sysadmin_refused_400",
        "msg_role_sysadmin",
        # Depth/nesting — declared caps bound, declared generosity holds.
        "chat_msgs_over",
        "chat_tools_over",
        "resp_input_over",
        "chat_content_part_100deep_ok",
        "resp_input_item_100deep_ok",
        # Structure — shapes refuse or accept as pinned.
        "chat_messages_empty",
        "chat_messages_dict",
        "resp_input_dict",
        "resp_input_string_ok",
        "chat_tool_calls_malformed_400",
        "chat_tools_dup_names_validated_then_501",
        # Cross-field rules.
        "chat_stream_plus_store_false_ok",
        "chat_tool_choice_fn_without_tools",
        "chat_rf_no_name_validates_then_output_502",
        "resp_prev_and_conversation",
        "chat_temp_and_top_p_ok",
        "resp_bg_nostore_400",
        "resp_bg_stream_streams",
        "chat_stream_options_without_stream_ok",
        "chat_max_tokens_conflict",
        # Metadata — 16/64/512 pinned, route-level 400 on vector stores,
        # honest unbounded pin on eval specs.
        "chat_metadata_17",
        "chat_metadata_key_65",
        "chat_metadata_value_int",
        "vs_metadata_17_refused_400",
        "espec_metadata_unbounded_17_accepted",
        # Unknown fields — allow swallows, forbid refuses incl. forges.
        "chat_extra_swallowed_ok",
        "harness_extra_forbidden",
        "espec_extra_forbidden",
        "harness_served_model_forge",
        # Content edges — verbatim execution, honest body cap.
        "chat_content_unicode_ok",
        "chat_content_nul_ok",
        "chat_content_ctrl_char_ok",
        "chat_content_ws_only_ok",
        "chat_body_over_1mb_413",
        "chat_body_just_under_1mb_ok",
        "body_nonjson_422",
        "body_truncated_422",
        "body_json_array_422",
        # Batch-line granularity vs whole-submit refusal.
        "batch_field_violation_errored_row",
        "batch_shape_error_whole_400",
        "batch_completes_with_errored_lines",
        "batch_no_row_is_5xx",
        "msgb_bad_params_whole_422",
        # Error shape — every refusal carries the declared envelope.
        "err_no_bare_500",
        "err_every_5xx_declared",
        "err_openai_422_envelope",
        "err_harness_422_detail_list",
        "err_anthropic_422_envelope",
        "err_capability_501_envelope",
        "err_translation_400_envelope",
        # Validity invariant — refused work never reaches the model.
        "inv_refusal_burns_no_backend",
        "inv_refusal_stores_no_record",
        "inv_refusal_no_served_calls",
        "inv_refusal_bills_uses",
        "inv_ok_call_stores_record",
        "inv_store_false_404",
        # Defect fixed this lane — an empty inner batch conversation used
        # to reach the backend and crash the call site (500).
        "cb_inner_empty_422",
        "cb_inner_empty_mixed_422",
        # The committed-receipt discipline audits itself.
        "verify_committed_receipt_valid_true",
    }
)


@pytest.fixture(scope="module")
def measured() -> dict[str, Any]:
    return audit.schema_audit()


def test_contract_probes_hold(measured: dict[str, Any]) -> None:
    assert len(measured) == 372
    assert all(value is True for value in measured.values()), measured


def test_named_pins_cover_every_validation_surface(measured: dict[str, Any]) -> None:
    assert measured.keys() >= _NAMED_PINS
    for name in sorted(_NAMED_PINS):
        assert measured[name] is True, name


@pytest.mark.parametrize("results", [{}, {"probe": False}, {"probe": 1}, {"probe": None}])
def test_receipt_refuses_empty_or_nonliteral_success(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, Any]
) -> None:
    monkeypatch.setattr(audit, "schema_audit", lambda: results)
    receipt = audit.schema_audit_bench()
    assert receipt["claim"]["ok"] is False
    assert receipt["claim"]["results"] == results
    assert receipt["interpretation"].startswith("SCHEMA AUDIT DEFECT:")
    assert verify_receipt_payload(receipt)["valid"] is True


def test_measured_receipt_verifies_without_rerunning(
    measured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "schema_audit", lambda: measured)
    receipt = audit.schema_audit_bench()
    assert receipt["claim"]["ok"] is True
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert verify_receipt_payload(receipt)["valid"] is True
    assert receipt == audit.schema_audit_bench()


def test_committed_receipt_still_verifies() -> None:
    import json
    from pathlib import Path

    path = Path("receipts/fx1_schema_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []
