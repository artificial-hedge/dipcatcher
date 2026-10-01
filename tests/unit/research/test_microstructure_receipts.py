"""Receipt contracts reject freshly resealed, internally inconsistent evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from quant_fund.research.microstructure_receipts import (
    MICROSTRUCTURE_RECEIPT_CONTRACTS,
    microstructure_receipt_contract_errors,
)
from quant_fund.research.receipt_v2 import (
    build_receipt_v2,
    seal_receipt,
    verify_receipt_payload,
)
from quant_fund.research.script_receipts import SCRIPT_RECEIPT_CONTRACTS

RECEIPTS = Path(__file__).resolve().parents[3] / "receipts"

# Each schema gets a concrete mutation of a count, algebraic relationship,
# selected summary, bounded statistic, or comparison claim. Resealing ensures
# that these failures come from the contract rather than mere byte tampering.
CASES: tuple[tuple[str, tuple[str | int, ...], object, str], ...] = (
    ("abc_calibrate_amzn", ("abc", "accepted", 0, "distance"), 9.0, "accepted_distance"),
    ("cancel_cluster_amzn", ("real", "buy_side", "lift_2s"), 99.0, "cancel_lift"),
    ("deep_microprice_amzn", ("real", "best_depth"), 1, "best_depth"),
    ("depth_consumption_amzn", ("real", "n_with_depth"), 1, "n_with_depth"),
    ("event_burst_amzn", ("real", "all", "burstiness_B"), 0.0, "burstiness_B"),
    ("event_granger_amzn", ("real", "submit->exec", "peak_lag_s"), 0.3, "peak_lag_s"),
    ("event_matrix_amzn", ("sim_arms", "iid", "steps_per_s"), 99.0, "steps_per_s"),
    ("exec_cost_real_amzn", ("real", "size_buckets", "qty_1_100", "n"), 1000000, "size_buckets"),
    ("exec_cost_split", ("claims", "correlated_flow_raises_cost"), False, "cost_claim"),
    ("glft_bench_synth", ("mc_check", "abs_diff"), 2.0, "mc_abs_diff"),
    ("hawkes_mv", ("real", "rho_branching"), 0.95, "rho_branching"),
    ("hawkes_real_amzn", ("real", "loglik_gain_vs_poisson"), 0.0, "loglik_gain"),
    ("hidden_depth_amzn", ("real", "hidden_trade_share"), 0.8, "hidden_trade_share"),
    ("impact_instant_amzn", ("real", "mean_signed_dmid_ticks"), 100.0, "signed_exceeds"),
    ("intraday_shape_amzn", ("real", "activity_u_shape"), 99.0, "activity_u_shape"),
    ("lob_exec_amzn", ("cells", "small_twap5", "is_total_ticks_mean"), 1000.0, "mean_outside"),
    ("lob_resilience_amzn", ("real", "ask", "n_recovered"), 1, "n_depletions"),
    ("marketable_limit_amzn", ("real", "share", "inside"), 0.3, "simplex"),
    ("metaorder_detect_amzn", ("real", "n_metaorders"), 1000000, "n_metaorders"),
    ("mid_jump_amzn", ("real", "move_share"), 0.8, "move_share"),
    ("order_lifetime_amzn", ("real", "n_orders_tracked"), 1, "n_orders_tracked"),
    ("order_revision_amzn", ("real", "toward_mid_share"), 0.8, "revision_direction"),
    ("post_trade_drift_amzn", ("real", "all", "n"), 1, "drift_partition"),
    ("price_improvement_amzn", ("real", "n_priced"), 1000000, "n_priced"),
    ("propagator_real_amzn", ("table", "ratio_real_over_iid@1"), 0.0, "response_ratio"),
    ("quote_place_amzn", ("real", "n_submissions"), 1, "dist_hist_ticks"),
    ("round_lot_amzn", ("real", "n_trades"), 1, "size_hist"),
    ("sign_autocorr_real_amzn", ("real", "n_positive_lags"), 1, "n_positive_lags"),
    ("sign_predict_amzn", ("real", "continuation", "1", "n"), 1, "continuation_k1"),
    ("sim_real_ledger_amzn", ("table", "sign_lag1", "calm_over_real"), 99.0, "ledger_ratio"),
    ("split_flow_bench", ("arms", 0, "mo_fraction"), 0.8, "mo_fraction"),
    ("spread_dynamics_amzn", ("real", "tight_share_le2"), 0.8, "tight_share_le2"),
    ("spread_response_amzn", ("real", "by_dir", "pooled", "n"), 1, "response_partition"),
    ("stale_quote_amzn", ("real", "age_bins", 0, "share"), 0.8, "age_bin_share"),
    ("streak_stats_amzn", ("real", "mean_run"), 99.0, "mean_run"),
    ("tape_digest_amzn", ("arm_coverage_score", "iid"), 99, "arm_coverage_score"),
    ("tick_rule_amzn", ("real", "quote_rule", "n"), 1, "rule_n"),
    ("vol_signature_amzn", ("real", "rv_fine_to_coarse_ratio"), 0.1, "rv_ratio"),
    ("vpin_amzn", ("real", "vpin_fwd_vol_corr"), 1.1, "vpin_fwd_vol_corr"),
)


def _load(name: str) -> dict[str, Any]:
    body: dict[str, Any] = json.loads((RECEIPTS / f"{name}.json").read_text())
    return body


def _mutate(body: dict[str, Any], path: tuple[str | int, ...], value: object) -> None:
    cursor: Any = body
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


def test_all_registered_schemas_have_a_deep_mutation_case() -> None:
    schemas = {_load(name)["schema"] for name, _, _, _ in CASES}
    assert schemas == MICROSTRUCTURE_RECEIPT_CONTRACTS.keys()
    assert schemas <= SCRIPT_RECEIPT_CONTRACTS.keys()


@pytest.mark.parametrize("name,path,value,error", CASES, ids=[case[0] for case in CASES])
def test_original_embedded_consistency_and_seal(
    name: str, path: tuple[str | int, ...], value: object, error: str
) -> None:
    original = _load(name)
    assert microstructure_receipt_contract_errors(original) == []
    assert verify_receipt_payload(original)["errors"] == []


@pytest.mark.parametrize("name,path,value,error", CASES, ids=[case[0] for case in CASES])
def test_forged_detail_fails_under_a_fresh_seal(
    name: str, path: tuple[str | int, ...], value: object, error: str
) -> None:
    forged = copy.deepcopy(_load(name))
    _mutate(forged, path, value)
    result = verify_receipt_payload(seal_receipt(forged))
    assert result["digest_convention"] is not None
    assert not any("seal" in item for item in result["errors"])
    assert any(error in item for item in result["errors"]), result["errors"]
    assert result["valid"] is False


@pytest.mark.parametrize("name,path,value,error", CASES, ids=[case[0] for case in CASES])
def test_schema_and_kind_renaming_do_not_evade_the_known_contract(
    name: str, path: tuple[str | int, ...], value: object, error: str
) -> None:
    body = _load(name)
    renamed_schema = seal_receipt({**body, "schema": "unknown.v1"})
    assert "schema:rederive_mismatch" in verify_receipt_payload(renamed_schema)["errors"]
    renamed_kind = seal_receipt({**body, "kind": "unknown"})
    assert "kind:rederive_mismatch" in verify_receipt_payload(renamed_kind)["errors"]


def test_wrapping_a_forgery_in_v2_preserves_the_inner_contract() -> None:
    inner = _load("vol_signature_amzn")
    inner["real"]["rv_fine_to_coarse_ratio"] = 99.0
    inner.pop("receipt_sha256")
    wrapped = seal_receipt(
        build_receipt_v2(
            kind="contract_test",
            data_label="MIXED",
            dataset={"source": "unverified_embedded_summary"},
            params={"synthetic_mutation": True},
            code_files=(Path(__file__),),
            verdict="fail",
            payload=inner,
        )
    )
    assert any("rv_ratio" in error for error in verify_receipt_payload(wrapped)["errors"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("research_only", False),
        ("research_only", 1),
        ("data_label", "REAL"),
        ("git_revision", "not_a_revision"),
        ("live_pnl_claim", True),
    ],
)
def test_synthetic_lane_honesty_and_provenance_fields(key: str, value: object) -> None:
    body = _load("exec_cost_split")
    body[key] = value
    assert verify_receipt_payload(seal_receipt(body))["valid"] is False


def test_bools_cannot_impersonate_counts_or_statistics() -> None:
    body = _load("vpin_amzn")
    body["real"]["n_buckets"] = True
    assert any("count_domain" in error for error in microstructure_receipt_contract_errors(body))
    body = _load("vpin_amzn")
    body["real"]["vpin_mean"] = True
    assert any("fraction_domain" in error for error in microstructure_receipt_contract_errors(body))


def test_malformed_known_schema_is_not_seal_only() -> None:
    for schema in MICROSTRUCTURE_RECEIPT_CONTRACTS:
        body = {"schema": schema, "research_only": True}
        assert microstructure_receipt_contract_errors(body)


def test_unknown_unregistered_schema_is_not_claimed_as_contract_covered() -> None:
    assert "unknown.v1" not in MICROSTRUCTURE_RECEIPT_CONTRACTS
    assert microstructure_receipt_contract_errors({"schema": "unknown.v1", "kind": "unknown"}) == []


def test_unavailable_rejected_draws_and_child_receipts_are_not_reexecuted() -> None:
    # The ABC median and tape-digest source hashes are external attestations.
    # A consistent change to such absent details cannot be independently
    # disproved here; this explicitly pins the limited scope of verification.
    abc = _load("abc_calibrate_amzn")
    abc["abc"]["median_draw_distance"] = 10.0
    assert microstructure_receipt_contract_errors(abc) == []
    digest = _load("tape_digest_amzn")
    digest["lanes"]["hawkes"]["receipt_sha256"] = "a" * 64
    assert microstructure_receipt_contract_errors(digest) == []
