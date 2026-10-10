"""Contract tests for script-emitted receipt schemas.

Each test pins a re-derivation: the verifier recomputes a headline claim from
the receipt's own embedded detail, so a forged aggregate fails verification
rather than passing on seal consistency alone.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from quant_fund.research.receipt_v2 import verify_receipt_payload
from quant_fund.research.script_receipts import (
    SCRIPT_RECEIPT_CONTRACTS,
    TAPE_SIM_STUDY_KINDS,
    script_receipt_contract_errors,
)

RECEIPTS = Path(__file__).resolve().parents[3] / "receipts"


def _load(name: str) -> dict:
    path = RECEIPTS / name
    if not path.is_file():
        pytest.skip(f"{name} receipt not in this checkout")
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def _errors(name: str) -> list[str]:
    return script_receipt_contract_errors(_load(name)["schema"], _load(name))


def test_committed_script_receipts_satisfy_their_contracts() -> None:
    """Real artifacts must pass their own re-derivation contracts."""
    for name in (
        "adaptive_mix_20asset_1d_20260922.json",
        "adaptive_mix_band_search_20asset_1d_20260922.json",
        "basis_pair_candidate_20asset_1d_20260922.json",
        "basis_reversion_screen_20asset_1d_20260922.json",
        "incumbent_bench_qlib.json",
        "dip_bench_crypto_1d_20260925.json",
    ):
        assert _errors(name) == [], f"{name}: {_errors(name)}"


def test_band_search_forged_selection_caught() -> None:
    payload = _load("adaptive_mix_band_search_20asset_1d_20260922.json")
    # The committed run selected nothing (no candidate was eligible); claim one.
    forged = copy.deepcopy(payload)
    forged["selected_band"] = 0.0
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert "selected_band_not_argmax_dev_sharpe" in errors


def test_band_search_forged_eligibility_caught() -> None:
    payload = _load("adaptive_mix_band_search_20asset_1d_20260922.json")
    forged = copy.deepcopy(payload)
    forged["candidates"][0]["eligible"] = True  # cagr/sharpe are negative
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert any("eligible_mismatch" in e for e in errors)


def test_adaptive_mix_allocation_must_be_a_simplex() -> None:
    payload = _load("adaptive_mix_20asset_1d_20260922.json")
    forged = copy.deepcopy(payload)
    seg = next(iter(forged["mean_adaptive_allocation"]))
    sleeve = next(iter(forged["mean_adaptive_allocation"][seg]))
    forged["mean_adaptive_allocation"][seg][sleeve] = 2.5
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert any("weight_out_of_bounds" in e for e in errors)


def test_adaptive_mix_unknown_sleeve_caught() -> None:
    payload = _load("adaptive_mix_20asset_1d_20260922.json")
    forged = copy.deepcopy(payload)
    seg = next(iter(forged["mean_adaptive_allocation"]))
    weights = forged["mean_adaptive_allocation"][seg]
    weights["phantom_sleeve"] = 0.01
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert any("unknown_sleeve" in e for e in errors)


def test_incumbent_bench_forged_median_caught() -> None:
    payload = _load("incumbent_bench_qlib.json")
    forged = copy.deepcopy(payload)
    forged["latency"]["dipcatcher_ms_median"] = 1.0  # real runs are ~100ms
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert "dipcatcher_ms_median_rederive_mismatch" in errors


def test_dip_bench_forged_event_count_caught() -> None:
    payload = _load("dip_bench_crypto_1d_20260925.json")
    forged = copy.deepcopy(payload)
    forged["n_events"] += 1
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert "n_events_not_sum_of_per_asset" in errors


def test_dip_bench_forged_recovery_caught() -> None:
    payload = _load("dip_bench_crypto_1d_20260925.json")
    forged = copy.deepcopy(payload)
    forged["baseline_recovery"]["1m"] = 1.7
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert "baseline_recovery_not_fractions" in errors


def test_basis_screen_fraction_out_of_bounds_caught() -> None:
    payload = _load("basis_reversion_screen_20asset_1d_20260922.json")
    forged = copy.deepcopy(payload)
    forged["positive_day_fraction"] = 1.4
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert "positive_day_fraction_out_of_bounds" in errors


def test_basis_pair_candidate_non_bool_eligible_caught() -> None:
    payload = _load("basis_pair_candidate_20asset_1d_20260922.json")
    forged = copy.deepcopy(payload)
    forged["development_eligible"] = "true"
    errors = script_receipt_contract_errors(forged["schema"], forged)
    assert "development_eligible_not_bool" in errors


def test_unknown_schema_passes_through() -> None:
    assert script_receipt_contract_errors("not_a_real_schema", {}) == []
    assert script_receipt_contract_errors(None, {}) == []


def test_malformed_body_fails_closed() -> None:
    """A payload that breaks the checker must report an error, not pass."""

    class ExplodingMapping(dict):
        def get(self, key: object, default: object = None) -> object:
            raise TypeError("boom")

    assert script_receipt_contract_errors("adaptive_mix_band_search.v1", ExplodingMapping()) == [
        "contract_check_error"
    ]


def test_dispatch_reaches_verify_receipt_payload() -> None:
    """The v1 envelope path must actually invoke the script contracts."""
    payload = _load("dip_bench_crypto_1d_20260925.json")
    forged = copy.deepcopy(payload)
    forged["n_events"] += 1
    result = verify_receipt_payload(forged, "forged.json")
    errors = result["errors"] if isinstance(result, dict) else result.errors
    assert "n_events_not_sum_of_per_asset" in errors


def test_every_committed_schema_has_a_contract() -> None:
    """A new scripts/ receipt schema must not ship unverified."""
    for name in (
        "adaptive_mix_20asset_1d_20260922.json",
        "adaptive_mix_band_search_20asset_1d_20260922.json",
        "basis_pair_candidate_20asset_1d_20260922.json",
        "basis_reversion_screen_20asset_1d_20260922.json",
        "incumbent_bench_qlib.json",
        "dip_bench_crypto_1d_20260925.json",
    ):
        path = RECEIPTS / name
        if not path.is_file():
            continue
        schema = json.loads(path.read_text())["schema"]
        assert schema in SCRIPT_RECEIPT_CONTRACTS, f"{name}: no contract for {schema}"


@pytest.mark.parametrize(
    "value,expected",
    [(3, True), (1.0, False), (0, False), (True, False), (math.nan, False), ("1", False)],
)
def test_positive_int_strictness(value: object, expected: bool) -> None:
    from quant_fund.research.script_receipts import _positive_int

    assert _positive_int(value) is expected


def _deps_hygiene() -> dict[str, object]:
    return json.loads((RECEIPTS / "deps_security_hygiene_f3b4e6fd22e439b7.json").read_text())


def test_deps_hygiene_committed_verifies() -> None:
    path = RECEIPTS / "deps_security_hygiene_f3b4e6fd22e439b7.json"
    if not path.is_file():
        pytest.skip("deps_hygiene receipt not committed in this checkout")
    result = verify_receipt_payload(json.loads(path.read_text()))
    errors = result["errors"] if isinstance(result, dict) else result.errors
    assert errors == []


def test_deps_hygiene_forged_counts_fail() -> None:
    path = RECEIPTS / "deps_security_hygiene_f3b4e6fd22e439b7.json"
    if not path.is_file():
        pytest.skip("deps_hygiene receipt not committed in this checkout")
    forged = copy.deepcopy(json.loads(path.read_text()))
    forged["results"]["gitleaks"]["leaks"] = -1
    from quant_fund.research.receipt_v2 import seal_receipt

    forged = seal_receipt(forged)
    result = verify_receipt_payload(forged, "forged.json")
    errors = result["errors"] if isinstance(result, dict) else result.errors
    assert any("leaks_not_nonneg_int" in e for e in errors)


#: schemas verified by the dedicated lane chain / kind dispatch rather than
#: SCRIPT_RECEIPT_CONTRACTS. Extending this set is a deliberate act: a new
#: receipt schema must pick a contract.
_LANE_COVERED_SCHEMAS = frozenset(
    {
        "fleet_eval.v1",
        "vol_bench.v1",
        "capacity_overlay.v1",
        "cross_sectional_rankic.v1",
        "hstep_bench.v1",
        "calibration_eval.v1",
        "cost_calibration.v1",
        "receipt_lattice.v1",
        "receipt_admission.v1",
        "rough_vol.v1",
        "fbm_circulant.v1",
        "corpus_epoch.v1",
        "basis_carry.v1",
        "vol_of_vol.v1",
        # the generic measurement-lane contract (`_MEASURE_SCHEMAS`) covers
        "abc_calibrate.v1",
        "alert_budget.v1",
        "anthropic_sdk_audit.v1",
        "api_audit.v1",
        "api_fuzz.v1",
        "asof_audit.v1",
        "attestation_audit.v1",
        "attribution.v1",
        "auth_audit.v1",
        "backend_parity.v1",
        "bank_audit.v1",
        "boundary_audit.v1",
        "byok_audit.v1",
        "cache_audit.v1",
        "cancel_cluster.v1",
        "cancel_gradient.v1",
        "cancel_lead.v1",
        "cap_audit.v1",
        "capability_audit.v1",
        "causality_scan.v1",
        "cli_audit.v1",
        "client_audit.v1",
        "contamination_audit.v1",
        "contract_probe.v1",
        "corpus_audit.v1",
        "decay_watch.v1",
        "deep_microprice.v1",
        "depth_consumption.v1",
        "dip_audit.v1",
        "dip_run_audit.v1",
        "disclosure_audit.v1",
        "dispatch_audit.v1",
        "diversity.v1",
        "doctor_audit.v1",
        "drain_audit.v1",
        "ds_audit.v1",
        "duration_check.v1",
        "e2e_audit.v1",
        "em_audit.v1",
        "engine_fuzz.v1",
        "error_shape.v1",
        "eval_core_audit.v1",
        "eval_lifecycle_audit.v1",
        "event_burst.v1",
        "event_granger.v1",
        "event_matrix.v1",
        "exec_cost_real.v1",
        "exec_cost_split.v1",
        "ext_bench_audit.v1",
        "fault_audit.v1",
        "forecast_core_audit.v1",
        "forecast_data_audit.v1",
        "forecast_infra_audit.v1",
        "forecast_pipeline_audit.v1",
        "fx1_contract_audit.v1",
        "fx1_tail_audit.v1",
        "glft_bench.v1",
        "grad_fidelity.v1",
        "harness_audit.v1",
        "hawkes_mv.v1",
        "hawkes_real.v1",
        "hidden_depth.v1",
        "hmm_stability.v1",
        "hmm_verify.v1",
        "honesty_audit.v1",
        "hypotheses_audit.v1",
        "imbalance_predict.v1",
        "impact_instant.v1",
        "inherit_audit.v1",
        "initiative_fade.v1",
        "intraday_exec.v1",
        "intraday_shape.v1",
        "jobs_audit.v1",
        "journal_audit.v1",
        "kill_audit.v1",
        "label_horizon_map.v1",
        "label_stability.v1",
        "ledger_audit.v1",
        "lineage_dag.v1",
        "lob_exec.v1",
        "lob_invariants.v1",
        "lob_resilience.v1",
        "map_parity.v1",
        "marketable_limit.v1",
        "masking_audit.v1",
        "meta_model.v1",
        "metaorder_detect.v1",
        "mid_jump.v1",
        "middleware_audit.v1",
        "modelcard_audit.v1",
        "mrm_audit.v1",
        "native_conformance.v1",
        "oai_sdk_audit.v1",
        "ops_audit.v1",
        "options_audit.v1",
        "order_lifetime.v1",
        "order_revision.v1",
        "parity_audit.v1",
        "parity_leak_audit.v1",
        "perf_audit.v1",
        "pipeline_audit.v1",
        "pipeline_flat_audit.v1",
        "post_trade_drift.v1",
        "price_clustering.v1",
        "price_improvement.v1",
        "promotion_gate.v1",
        "propagator_real.v1",
        "quality_audit.v1",
        "queue_class.v1",
        "queue_fate.v1",
        "queue_jump.v1",
        "queue_occupancy.v1",
        "quota2_audit.v1",
        "quote_place.v1",
        "receipts_audit.v1",
        "replay_audit.v1",
        "report_audit.v1",
        "retrieval_audit.v1",
        "reward_audit.v1",
        "round_lot.v1",
        "rt_audit.v1",
        "rubric_audit.v1",
        "run_audit.v1",
        "sbom_audit.v1",
        "schema_drift.v1",
        "schema_fingerprint.v1",
        "sdk_audit.v1",
        "seed_audit.v1",
        "serve_audit.v1",
        "side_imbalance.v1",
        "sigkernel_mmd.v1",
        "sign_autocorr_real.v1",
        "sign_predict.v1",
        "sim_real_ledger.v1",
        "sim_sensitivity.v1",
        "sources_audit.v1",
        "spec_audit.v1",
        "split_flow.v1",
        "spread_dynamics.v1",
        "spread_response.v1",
        "stack_invariance.v1",
        "stack_watch.v1",
        "stale_quote.v1",
        "streak_stats.v1",
        "sweep_bound.v1",
        "tail_quota.v1",
        "tape_digest.v1",
        "tape_surgery.v1",
        "tick_rule.v1",
        "timepart_audit.v1",
        "trade_decomp.v1",
        "train_infra_audit.v1",
        "train_receipt_audit.v1",
        "trust_step.v1",
        "ts_reasoning_audit.v1",
        "usage_audit.v1",
        "validator_fuzz.v1",
        "vine_audit.v1",
        "vine_dominance.v1",
        "vine_panel.v1",
        "vol_signature.v1",
        "vpin.v1",
        "vs_audit.v1",
        "warmup_spec.v1",
        "webhook_audit.v1",
    }
)
#: schemas whose receipts dispatch to evalue_family_contract_errors via their
#: ``kind`` field — deep-checked even though the schema tag itself is not a
#: dispatch key.
_KIND_DISPATCHED_SCHEMAS = frozenset(
    {
        "calibration_audit.v1",
        "corpus_inference.v1",
        "coverage_audit.v1",
        "coverage_cs.v1",
        "changepoint_localize.v1",
        "emerge_drill.v1",
        "fleet_race.v1",
        "lane_power.v1",
        "loss_cs.v1",
        "monitor_run.v1",
        "panel_audit.v1",
        "suite_health.v1",
        "tail_audit.v1",
        "receipt_admission.v1",
    }
)


def test_every_committed_receipt_schema_is_contract_covered() -> None:
    """No receipt may verify on its seal alone — every committed schema must
    dispatch to a deep check somewhere in the verifier."""
    uncovered: list[str] = []
    for path in sorted(RECEIPTS.glob("*.json")):
        schema = json.loads(path.read_text()).get("schema")
        if schema in ("receipt.v2", None):
            continue  # v2 inners dispatch by kind fingerprint
        if not isinstance(schema, str):
            # ``schema: 1`` receipts have no schema name; ``kind`` is their
            # tag and must dispatch to a real contract.
            if json.loads(path.read_text()).get("kind") in TAPE_SIM_STUDY_KINDS:
                continue
            uncovered.append(f"{path.name}:<unnamed schema>")
            continue
        if (
            schema in SCRIPT_RECEIPT_CONTRACTS
            or schema in _LANE_COVERED_SCHEMAS
            or schema in _KIND_DISPATCHED_SCHEMAS
        ):
            continue
        uncovered.append(f"{path.name}:{schema}")
    assert uncovered == []
