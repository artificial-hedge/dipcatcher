"""Tests for fx1.serve.selftest_audit — the deploy-gate honesty battery."""

from __future__ import annotations

from typing import Any

import pytest

from fx1.serve import selftest_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload

_NAMED_PINS = frozenset(
    {
        # report contract
        "empty_report_not_ok",
        "one_failure_flips_ok",
        "note_records_verdict",
        "as_dict_json_roundtrip",
        "as_dict_lists_every_check",
        "crashed_check_is_named_failure",
        "crash_does_not_mask_later_checks",
        "falsy_result_fails_with_got",
        "expect_mismatch_reports_got",
        # local golden path
        "local_report_ok",
        "local_mode_labeled",
        "local_check_set",
        "local_checks_unique_named",
        "local_failures_named",
        "local_as_dict_honest",
        "local_deterministic",
        # env/state isolation
        "env_run_still_ok",
        "moonshot_key_restored",
        "byok_env_restored",
        "api_key_env_restored",
        "byok_binds_own_stub",
        "byok_ambient_env_restored",
        "concurrent_runs_both_ok",
        "concurrent_same_check_vector",
        "concurrent_env_clean_after",
        "state_dir_run_ok",
        "state_dir_adds_restart_check",
        "state_dir_check_count",
        "state_writes_confined",
        "state_nonempty_journals",
        "state_job_journal_written",
        "selftest_writes_no_report_artifact",
        # remote mode
        "remote_mode_labeled",
        "remote_clean_ok",
        "remote_check_set",
        "remote_methods_readonly",
        "remote_posts_advisory_or_refused",
        "remote_no_mutating_route_2xx",
        "remote_state_untouched",
        "remote_deterministic",
        "remote_no_key_cannot_pass",
        "remote_no_key_names_refusals",
        "remote_no_key_health_still_public",
        "remote_no_key_skips_auth_check",
        "remote_wrong_key_fails",
        "remote_wrong_key_gate_still_honest",
        "remote_unauthed_caught",
        "remote_unauthed_advisories_pass",
        "remote_open_deployment_advisories_ok",
        "remote_open_auth_unproven",
        "remote_dead_not_ok",
        "remote_dead_failures_named",
        "remote_dead_returns_bounded",
        "remote_500s_not_ok",
        "remote_500s_failures_named",
        # gate parity
        "parity_version_field",
        "parity_missing_key_is_auth_error",
        "parity_wrong_key_is_auth_error",
        "parity_bogus_backend_maps_valueerror",
        "parity_complete_byok_works",
        "parity_receipt_verifies",
        "parity_advisory_surfaces",
        # bench unit honesty
        "bench_clean_zero_errors",
        "bench_measured_requests_exact",
        "bench_warmup_unmeasured",
        "bench_latency_card_ordered",
        "bench_latency_wall_measured",
        "bench_prompt_hashed_not_embedded",
        "bench_mode_recorded",
        "bench_tokens_accounted",
        "bench_models_backends_reported",
        "bench_refusal_storm_measured_error",
        "bench_refusal_rate_honest",
        "bench_refusal_class_honest",
        "bench_refusal_returns_bounded",
        "bench_histogram_per_class",
        "bench_mixed_ok_counted_honestly",
        "bench_bounds_n_low",
        "bench_bounds_n_high",
        "bench_bounds_concurrency",
        "bench_bounds_warmup",
        "bench_bounds_max_tokens",
        "bench_bounds_timeout",
        "bench_bounds_prompt_empty",
        "bench_bounds_prompt_long",
        "bench_bounds_mode",
        "bench_bounds_check_before_spend",
        # CLI surface
        "cli_local_exit0",
        "cli_report_ok_field",
        "cli_report_lists_checks",
        "cli_dead_remote_exit2",
        "cli_dead_report_names_failures",
        "cli_live_remote_exit0",
        "cli_wrong_key_exit2",
        "cli_bench_live_exit0",
        "cli_bench_live_measured",
        "cli_bench_dead_exit1",
        "cli_bench_dead_histogram_honest",
        "cli_bench_wrong_key_exit1",
        "cli_bench_wrong_key_histogram",
        "cli_bench_drained_exit1",
        "cli_bench_drained_histogram",
        "cli_bench_receipt_verifies",
        # doctor
        "doctor_exit0",
        "doctor_flags_honest",
        "doctor_json_serializable",
        "doctor_set_flag_honest",
        "doctor_never_leaks_value",
        "harness_doctor_absent",
    }
)

#: Probes that pin defects found by this lane (fixed in src/fx1/selftest.py).
_FORMER_DEFECTS = frozenset(
    {
        "moonshot_key_restored",  # popped env var was never restored
        "concurrent_runs_both_ok",  # parallel local runs raced on os.environ
        "concurrent_same_check_vector",
    }
)


@pytest.fixture(scope="module")
def measured() -> dict[str, Any]:
    return audit.selftest_audit()


def test_contract_probes_hold(measured: dict[str, Any]) -> None:
    assert len(measured) == 109
    assert all(value is True for value in measured.values()), measured


def test_named_pins_cover_the_deploy_gate(measured: dict[str, Any]) -> None:
    assert measured.keys() >= _NAMED_PINS
    for name in sorted(_NAMED_PINS):
        assert measured[name] is True, name


def test_former_defects_stay_fixed(measured: dict[str, Any]) -> None:
    for name in sorted(_FORMER_DEFECTS):
        assert measured[name] is True, name


@pytest.mark.parametrize("results", [{}, {"probe": False}, {"probe": 1}, {"probe": None}])
def test_receipt_refuses_empty_or_nonliteral_success(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, Any]
) -> None:
    monkeypatch.setattr(audit, "selftest_audit", lambda: results)
    receipt = audit.selftest_audit_bench()
    assert receipt["claim"]["ok"] is False
    assert receipt["claim"]["results"] == results
    assert receipt["interpretation"].startswith("SELFTEST AUDIT DEFECT:")
    assert verify_receipt_payload(receipt)["valid"] is True


def test_measured_receipt_verifies_without_rerunning(
    measured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "selftest_audit", lambda: measured)
    receipt = audit.selftest_audit_bench()
    assert receipt["claim"]["ok"] is True
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert verify_receipt_payload(receipt)["valid"] is True
    assert receipt == audit.selftest_audit_bench()


def test_committed_receipt_still_verifies() -> None:
    import json
    from pathlib import Path

    path = Path("receipts/fx1_selftest_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []
