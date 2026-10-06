"""Synthetic eval-gate probes and the audit's helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import evalgates_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def measured() -> dict[str, Any]:
    return audit.evalgates_audit()


def test_contract_probes_hold(measured: dict[str, Any]) -> None:
    assert len(measured) == 152
    assert all(value is True for value in measured.values()), measured


def test_score_contract_legs(measured: dict[str, Any]) -> None:
    for name in (
        "score_deterministic_bytes",
        "score_envelope_shape",
        "score_minimal_honesty_weight",
        "score_empty_total_zero",
        "score_full_components",
        "score_partial_attribution",
        "score_receipt_window_bound",
        "score_violation_200_not_refused",
        "score_violation_caps_negative",
        "score_violation_no_partial_credit",
        "score_violation_drops_loaded_credit",
        "score_live_claim_negative",
        "score_unlabeled_synthetic_negative",
        "score_labeled_synthetic_clean",
        "score_homoglyph_negative",
        "score_spaced_letters_negative",
        "score_fullwidth_negative",
        "score_list_independent_items",
        "score_list_matches_singletons",
        "score_128_items_ok",
        "score_129_items_422",
        "score_overlong_422",
        "score_extra_field_422",
        "score_identical_across_backend_configs",
    ):
        assert measured[name] is True, name


def test_gate_contract_legs(measured: dict[str, Any]) -> None:
    for name in (
        "gate_clean_ok",
        "gate_error_field_explicit_null",
        "gate_each_forbidden_token_refused",
        "gate_refusal_names_reason",
        "gate_error_names_token",
        "gate_profit_loss_spelling_refused",
        "gate_live_claim_refused",
        "gate_unlabeled_synthetic_refused",
        "gate_labeled_synthetic_ok",
        "gate_bare_token_ok",
        "gate_spelled_number_ok",
        "gate_homoglyph_refused",
        "gate_spaced_letters_refused",
        "gate_fullwidth_refused",
        "gate_verdict_surface_200",
        "gate_deterministic_bytes",
        "gate_overlong_422",
        "gate_extra_field_422",
        "gate_score_coherent_on_violation",
        "gate_score_coherent_on_clean",
    ):
        assert measured[name] is True, name


def test_diff_live_and_shape_legs(measured: dict[str, Any]) -> None:
    for name in (
        "diff_200",
        "diff_shape",
        "diff_comparable_same_suite_seed",
        "diff_unstamped_same_bank_false",
        "diff_fixed_listed",
        "diff_verdict_improved",
        "diff_tooluse_gate_unknown",
        "diff_sig_exact_n2",
        "diff_regressed_listed",
        "diff_regressed_verdict",
        "diff_mixed_both_listed",
        "diff_regressed_beats_fixed",
        "diff_significant_p05",
        "diff_identical_unchanged",
        "diff_identical_sig_p1",
        "diff_self_all_empty",
        "ts_gate_transition_opened",
        "ts_fixed_results_shape",
        "ts_domain_regressed_verdict",
        "ts_domain_gate_unchanged",
        "ret_fixed_question_id_shape",
        "ret_gate_opened",
    ):
        assert measured[name] is True, name


def test_comparability_and_injected_legs(measured: dict[str, Any]) -> None:
    for name in (
        "cross_seed_served_200",
        "cross_seed_not_comparable",
        "cross_seed_verdict_unknown",
        "cross_seed_significance_null",
        "cross_seed_transitions_dropped",
        "cross_suite_not_comparable",
        "bank_mismatch_flagged",
        "bank_stamp_match_same_bank",
        "bank_one_stamp_comparable",
        "inj_deltas_only_changed_shared_numeric",
        "inj_deltas_excludes_bool_and_onesided",
        "inj_tasks_only_lists",
        "inj_gate_closed_verdict_regressed",
        "inj_gate_opened_verdict_improved",
        "inj_gate_missing_unknown",
        "inj_regressed_beats_fix_and_gate",
        "inj_sig_n5_insufficient",
        "inj_sig_n1_p1",
        "inj_terminal_kinds_diffable",
        "diff_pure_store_read",
    ):
        assert measured[name] is True, name


def test_state_drain_auth_client_envelope_legs(measured: dict[str, Any]) -> None:
    for name in (
        "diff_unknown_base_404",
        "diff_404_names_base_first",
        "diff_404_enveloped",
        "diff_unknown_cand_404",
        "diff_running_409",
        "diff_queued_409",
        "diff_failed_no_report_409",
        "diff_wrong_verb_405",
        "diff_405_enveloped",
        "drain_latched",
        "drain_score_open",
        "drain_gate_open",
        "drain_score_violation_still_scored",
        "drain_diff_open",
        "drain_submit_refused_503",
        "drain_refusal_enveloped",
        "drain_cancel_terminal_state_verdict",
        "score_write_scope_200",
        "score_read_refused_403",
        "gate_write_scope_200",
        "gate_read_refused_403",
        "diff_read_scope_200",
        "diff_write_only_refused_403",
        "score_unauth_401",
        "gate_unauth_401",
        "diff_unauth_401",
        "client_check_text_clean_ok",
        "client_check_text_violation_refused",
        "client_score_matches_wire",
        "client_diff_matches_wire",
        "client_diff_404_keyerror",
        "client_diff_409_transport",
        "client_score_422_valueerror",
        "client_score_401_autherror",
        "sdk_check_text_twin",
        "sdk_score_twin",
        "sdk_eval_diff_keyerror",
        "env_404",
        "env_score_422",
        "env_gate_422",
        "env_401",
        "env_405",
        "restart_records_replayed",
        "restart_diff_served",
        "restart_verdict_identical",
    ):
        assert measured[name] is True, name


@pytest.mark.parametrize(
    "results",
    [{}, {"probe": True}, {"probe": False}, {"probe": 1}, {"probe": None}],
)
def test_receipt_refuses_empty_or_nonliteral_success(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, Any]
) -> None:
    monkeypatch.setattr(audit, "evalgates_audit", lambda: results)
    receipt = audit.evalgates_audit_bench()
    assert receipt["claim"]["ok"] is False
    assert receipt["claim"]["results"] == results
    assert receipt["interpretation"].startswith("EVALGATES AUDIT DEFECT:")
    assert verify_receipt_payload(receipt)["valid"] is True


def test_measured_receipt_verifies_without_rerunning(
    measured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "evalgates_audit", lambda: measured)
    receipt = audit.evalgates_audit_bench()
    assert receipt["claim"]["ok"] is True
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert verify_receipt_payload(receipt)["valid"] is True
    assert receipt == audit.evalgates_audit_bench()
    assert "TypeScript client runtime" in receipt["coverage"]["not_executed"]


def test_committed_receipt_matches_fresh_measurement(
    measured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    path = Path("receipts/fx1_evalgates_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["schema"] == "evalgates_audit.v1"
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []
    monkeypatch.setattr(audit, "evalgates_audit", lambda: measured)
    fresh = audit.evalgates_audit_bench()
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
        assert payload[field] == fresh[field], field


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
        client, _ = audit._client({"byok": lambda: audit._TooluseBackend({})})
        assert client.get("/health").status_code == 200
        executor = client.app.state.jobs_executor
        assert executor.submit(lambda: 3).result(timeout=1) == 3
    assert client.is_closed
    with pytest.raises(RuntimeError, match="shutdown"):
        executor.submit(lambda: None)
    assert list(operator.iterdir()) == [marker]
    assert marker.read_text() == "operator sentinel\n"


def test_tooluse_backend_replays_declared_plans() -> None:
    """The scripted backend replays the suite bank's own golden plan —
    step counting follows the assistant turns in the conversation."""
    import json

    backend = audit._TooluseBackend({})
    prompt = [{"role": "user", "content": "task [task_id: tooluse-pinball-grid]"}]
    first = json.loads(backend.complete(prompt))
    assert first == {"tool": "research", "args": {"bench": "pinball_grid"}}
    second = json.loads(
        backend.complete([*prompt, {"role": "assistant", "content": json.dumps(first)}])
    )
    assert "final" in second


def test_oracle_backend_overrides_only_scripted_ids() -> None:
    oracle = lambda messages: "canonical"  # noqa: E731 — fixture fn
    backend = audit._OracleBackend(oracle, {"ts-bait-00": "override"})
    hit = [{"role": "user", "content": "prompt [task_id: ts-bait-00]"}]
    miss = [{"role": "user", "content": "prompt [task_id: ts-ident-00]"}]
    assert backend.complete(hit) == "override"
    assert backend.complete(miss) == "canonical"


def test_mk_record_is_terminal_with_report() -> None:
    rec = audit._mk_record("r1", report={"results": []})
    assert rec.status == "succeeded" and rec.report == {"results": []}
    bare = audit._mk_record("r2", status="failed")
    assert bare.status == "failed" and bare.report is None
