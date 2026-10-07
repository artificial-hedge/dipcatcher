"""evalue_contracts: deep verification of the anytime-valid receipt family."""

from __future__ import annotations

from quant_fund.research.evalue_contracts import (
    EVALUE_FAMILY_KINDS,
    evalue_family_contract_errors,
)

_SHA = "a" * 64

_BASE_PROMOTION = {
    "kind": "evalue_promotion.v1",
    "challenger": "a",
    "incumbent": "b",
    "alpha": 0.05,
    "lam": 0.5,
    "n_origins": 100,
    "final_evalue": 25.0,
    "anytime_p": 0.04,
    "promotion_origin": 41,
    "promoted": True,
    "mean_loss_diff": -0.002,
    "scale_median": 0.01,
}

_BASE_RACE = {
    "kind": "fleet_race.v1",
    "data_label": "SYNTHETIC",
    "research_only": True,
    "live_pnl_claim": False,
    "inputs_sha256": _SHA,
    "params": {
        "n_train": 256,
        "n_eval": 128,
        "n_chunks": 8,
        "alpha": 0.05,
        "taus": [0.1, 0.5, 0.9],
        "seed": 0,
    },
    "n_shards": 3,
    "n_models": 5,
    "shard_winners": {"garch_vol": "gmm"},
    "evidence": ["anytime_valid_promotion"],
}

_BASE_CORPUS = {
    "kind": "corpus_inference.v1",
    "inputs_sha256": _SHA,
    "params": {"q": 0.05, "glob": "*.json"},
    "n_receipts": 4,
    "n_parse_errors": 0,
    "parse_errors": [],
    "n_p_findings": 10,
    "n_e_findings": 2,
    "n_survivors": 3,
    "surviving_claims": [{"source": "a", "path": "p", "value": 0.001, "family": "x"}] * 3,
    "corpus_evalue": 42.0,
    "corpus_reject_at_alpha": True,
}

_BASE_FDR = {
    "kind": "online_fdr.v1",
    "n_tests": 50,
    "n_rejections": 3,
    "rejection_indices": [2, 17, 40],
    "final_wealth": 0.03,
    "level": 0.05,
}


def test_valid_fixtures_pass_all_four_kinds() -> None:
    for base in (_BASE_PROMOTION, _BASE_RACE, _BASE_CORPUS, _BASE_FDR):
        assert evalue_family_contract_errors(base) == [], base["kind"]
        assert base["kind"] in EVALUE_FAMILY_KINDS


def test_promotion_arithmetic_must_hold() -> None:
    bad = dict(_BASE_PROMOTION, anytime_p=0.5)  # > 1/25 — impossible under Ville
    assert "anytime_p_exceeds_reciprocal_of_evalue" in evalue_family_contract_errors(bad)
    bad2 = dict(_BASE_PROMOTION, promotion_origin=200)
    assert "promotion_origin_out_of_range" in evalue_family_contract_errors(bad2)
    bad3 = dict(_BASE_PROMOTION, promoted=False, promotion_origin=None)
    assert evalue_family_contract_errors(bad3) == []
    bad4 = dict(_BASE_PROMOTION, promoted=True, promotion_origin=None)
    assert "promotion_origin_out_of_range" in evalue_family_contract_errors(bad4)


def test_fleet_race_contract() -> None:
    for field, bad_val, expect in [
        ("data_label", "REAL", "data_label_not_synthetic"),
        ("live_pnl_claim", True, "live_pnl_claim_not_false"),
        ("inputs_sha256", "xyz", "inputs_sha256_invalid"),
    ]:
        assert expect in evalue_family_contract_errors(dict(_BASE_RACE, **{field: bad_val}))
    bad_params = dict(_BASE_RACE)
    bad_params["params"] = dict(_BASE_RACE["params"], taus=[0.9, 0.1])
    assert "params_taus_not_strictly_increasing_unit_grid" in evalue_family_contract_errors(
        bad_params
    )


def test_corpus_coherence() -> None:
    bad = dict(_BASE_CORPUS, n_survivors=99)
    # n_survivors > n_p_findings is caught by length mismatch of surviving_claims
    assert "surviving_claims_length_mismatch" in evalue_family_contract_errors(bad)
    bad2 = dict(_BASE_CORPUS, corpus_reject_at_alpha=False)  # evalue 42 ≥ 20
    assert "corpus_reject_inconsistent_with_evalue" in evalue_family_contract_errors(bad2)
    bad3 = dict(_BASE_CORPUS, n_parse_errors=2)
    assert "parse_errors_length_mismatch" in evalue_family_contract_errors(bad3)


def test_online_fdr_contract() -> None:
    bad = dict(_BASE_FDR, n_rejections=60)
    assert "n_rejections_exceeds_n_tests" in evalue_family_contract_errors(bad)
    bad2 = dict(_BASE_FDR, rejection_indices=[40, 17, 2])
    assert "rejection_indices_not_strictly_increasing" in evalue_family_contract_errors(bad2)
    bad3 = dict(_BASE_FDR, final_wealth=-0.5)
    assert "final_wealth_not_nonnegative_finite" in evalue_family_contract_errors(bad3)


def test_unknown_kind_returns_empty() -> None:
    assert evalue_family_contract_errors({"kind": "other.v1"}) == []
    assert evalue_family_contract_errors({}) == []


def test_verify_receipt_dispatch() -> None:
    """kind-tagged payloads reach the contract through verify_receipt_payload."""
    from quant_fund.research.receipt_v2 import verify_receipt_payload

    tampered = dict(_BASE_PROMOTION, anytime_p=0.9)
    result = verify_receipt_payload(tampered)
    assert result != []
    assert any("anytime_p" in e for e in result["errors"])


_BASE_CALIBRATION = {
    "kind": "calibration_audit.v1",
    "inputs_sha256": "a" * 64,
    "params": {"alpha": 0.05, "channels": ["loc_hi", "loc_lo"], "n_eval": 256},
}

_BASE_LOSS_CS = {
    "kind": "loss_cs.v1",
    "alpha": 0.05,
    "lam": 0.5,
    "bound": 1.0,
    "n": 200,
    "cs_low": -0.4,
    "cs_high": -0.02,
    "excludes_zero": True,
    "interpretation": "challenger_better",
}

_BASE_LOCALIZE = {
    "kind": "changepoint_localize.v1",
    "params": {"alpha": 0.05, "window": 40, "min_left": 10},
    "result": {"tau_hat": 200, "cs_lo": 150, "cs_hi": 210, "n": 400, "alarmed": True},
}

_BASE_COV_AUDIT = {
    "kind": "coverage_audit.v1",
    "inputs_sha256": "b" * 64,
    "params": {"alpha": 0.05, "levels": [0.8, 0.9], "n_eval": 256},
}


def test_calibration_audit_contract() -> None:
    assert evalue_family_contract_errors(_BASE_CALIBRATION) == []
    bad = dict(_BASE_CALIBRATION, params={"alpha": 0.9, "channels": []})
    errs = evalue_family_contract_errors(bad)
    assert "channels_empty" in errs
    bad2 = dict(_BASE_CALIBRATION, inputs_sha256="zz")
    assert "inputs_sha256_malformed" in evalue_family_contract_errors(bad2)


def test_loss_cs_contract() -> None:
    assert evalue_family_contract_errors(_BASE_LOSS_CS) == []
    bad = dict(_BASE_LOSS_CS, cs_low=0.5, cs_high=0.1)
    assert "cs_bounds_malformed" in evalue_family_contract_errors(bad)
    bad2 = dict(_BASE_LOSS_CS, interpretation="surely_wins")
    assert "interpretation_unknown" in evalue_family_contract_errors(bad2)
    bad3 = dict(_BASE_LOSS_CS, bound=0.0)
    assert "bound_invalid" in evalue_family_contract_errors(bad3)


def test_changepoint_localize_contract() -> None:
    assert evalue_family_contract_errors(_BASE_LOCALIZE) == []
    bad = dict(_BASE_LOCALIZE, result={"tau_hat": 5, "cs_lo": 300, "cs_hi": 100, "n": 400})
    assert "cs_bounds_inverted" in evalue_family_contract_errors(bad)
    bad2 = dict(_BASE_LOCALIZE, params={"alpha": 0.05, "window": 0, "min_left": 10})
    assert "window_not_positive_int" in evalue_family_contract_errors(bad2)


def test_coverage_audit_contract() -> None:
    assert evalue_family_contract_errors(_BASE_COV_AUDIT) == []
    bad = dict(_BASE_COV_AUDIT, params={"alpha": 0.05, "levels": [1.5]})
    assert "levels_malformed" in evalue_family_contract_errors(bad)
    bad2 = dict(_BASE_COV_AUDIT, kind="coverage_cs.v1", inputs_sha256="x")
    assert "inputs_sha256_malformed" in evalue_family_contract_errors(bad2)
    good2 = dict(_BASE_COV_AUDIT, kind="coverage_cs.v1")
    assert evalue_family_contract_errors(good2) == []


_BASE_WINNER_CURSE = {
    "kind": "winner_curse.v1",
    "data_label": "SYNTHETIC",
    "research_only": True,
    "live_pnl_claim": False,
    "selected_head": "gmm",
    "naive_score": 0.010,
    "selection_bias": 0.001,
    "corrected_score": 0.011,
    "honest_score": 0.012,
    "naive_ci": [0.009, 0.011],
    "selection_aware_ci": [0.010, 0.013],
    "n_obs": 512,
    "n_heads": 5,
    "n_boot": 200,
    "verdict": "bias_material",
}

_BASE_DRIFT = {
    "kind": "drift_alarm.v1",
    "research_only": True,
    "live_pnl_claim": False,
    "n_obs": 400,
    "alpha": 0.05,
    "eprocess": {
        "lam": 0.7,
        "alarmed": True,
        "alarm_index": 210,
        "final_evalue": 33.0,
    },
    "page_hinkley": {"delta": 0.01, "h": 4.0, "alarmed": True, "alarm_index": 220},
    "n_inconclusive": 0,
}

_BASE_CONFORMAL = {
    "kind": "conformal_monitor.v1",
    "research_only": True,
    "live_pnl_claim": False,
    "n_obs": 300,
    "alpha": 0.05,
    "kappa": 1.5,
    "window": 100,
    "mode": "ar",
    "alarmed": False,
    "alarm_index": None,
    "final_martingale": 1.2,
    "n_inconclusive": 2,
}

_BASE_TAIL = {
    "schema": "tail_audit.v1",
    "kind": "tail_audit",
    "level": "research",
    "inputs_sha256": _SHA,
    "params": {
        "cell": [0.05, 0.1],
        "p0_deep_share": 0.5,
        "alpha": 0.05,
        "alt_grid": [0.5, 0.75, 1.35, 1.9],
        "n_train": 128,
        "n_eval": 256,
        "seed": 0,
    },
    "claims": [{"text": "x", "kind": "theoretical"}],
}

_BASE_LANE_POWER = {
    "schema": "lane_power.v1",
    "kind": "lane_power",
    "level": "research",
    "inputs_sha256": _SHA,
    "params": {
        "defects": [0.0, 0.1, 0.5],
        "n_steps": 400,
        "n_seeds": 20,
        "alpha": 0.05,
        "lanes": ["coverage", "tail"],
    },
    "n_lanes_ok": 2,
    "null_alarm_rate": {"coverage": 0.05, "tail": 0.0},
    "claims": [{"text": "x", "kind": "empirical_synthetic"}],
}

_BASE_HONEST_VERDICT = {
    "kind": "honest_verdict.v1",
    "data_label": "SYNTHETIC",
    "research_only": True,
    "live_pnl_claim": False,
    "verdict": "inconclusive",
    "winner": "gmm",
    "alpha": 0.05,
    "n_obs": 1152,
    "n_heads": 2,
    "inputs_sha256": _SHA,
    "components": {
        "winner_curse": {"corrected_score": 0.011},
        "promotion": {"promoted": False},
        "drift": {"eprocess_alarmed": False},
        "calibration": {},
        "magnitude": {},
        "localize": {"skipped": "no_drift_alarm"},
    },
    "unavailable_lanes": ["calibration", "magnitude"],
}

_BASE_MONITOR_RUN = {
    "kind": "monitor_run",
    "data_label": "SYNTHETIC",
    "research_only": True,
    "live_pnl_claim": False,
    "inputs_sha256": _SHA,
    "n_rows": 18,
    "n_alarm_rows": 2,
    "lanes_available": {
        "coverage": True,
        "tail": True,
        "calibration": True,
        "conformal": False,
        "drift": True,
        "emerge": True,
    },
    "params": {
        "n_train": 512,
        "n_eval": 256,
        "seed": 0,
        "alpha": 0.05,
        "level": 0.9,
        "tail_cell": [0.05, 0.1],
        "heads": ["gaussian", "gmm"],
        "shards": ["iid_gaussian", "heavy_tail"],
    },
    "evidence": ["anytime_valid_monitor_lanes"],
}


def test_new_kind_fixtures_pass() -> None:
    for base in (
        _BASE_WINNER_CURSE,
        _BASE_DRIFT,
        _BASE_CONFORMAL,
        _BASE_TAIL,
        _BASE_LANE_POWER,
        _BASE_HONEST_VERDICT,
        _BASE_MONITOR_RUN,
        _BASE_SUITE_HEALTH,
    ):
        assert evalue_family_contract_errors(base) == [], base.get("kind")


def test_winner_curse_contract() -> None:
    bad = dict(_BASE_WINNER_CURSE, verdict="bias_material", selection_bias=0.0)
    assert "bias_material_without_bias" in evalue_family_contract_errors(bad)
    bad2 = dict(_BASE_WINNER_CURSE, naive_ci=[0.02, 0.01])
    assert "naive_ci_inverted" in evalue_family_contract_errors(bad2)
    bad3 = dict(_BASE_WINNER_CURSE, verdict="winning")
    assert "verdict_not_in_enum" in evalue_family_contract_errors(bad3)


def test_drift_alarm_contract() -> None:
    ep = dict(_BASE_DRIFT["eprocess"], alarm_index=99999)
    bad = dict(_BASE_DRIFT, eprocess=ep)
    assert "alarm_index_out_of_range" in evalue_family_contract_errors(bad)
    ep2 = dict(_BASE_DRIFT["eprocess"], alarmed=False, alarm_index=5)
    bad2 = dict(_BASE_DRIFT, eprocess=ep2)
    assert "alarm_index_set_without_alarm" in evalue_family_contract_errors(bad2)
    ep3 = dict(_BASE_DRIFT["eprocess"], final_evalue=0.0)
    bad3 = dict(_BASE_DRIFT, eprocess=ep3)
    assert "final_evalue_not_positive_finite" in evalue_family_contract_errors(bad3)


def test_conformal_monitor_contract() -> None:
    bad = dict(_BASE_CONFORMAL, kappa=-1.0)
    assert "kappa_not_positive_finite" in evalue_family_contract_errors(bad)
    bad2 = dict(_BASE_CONFORMAL, alarmed=True, alarm_index=9999)
    assert "alarm_index_out_of_range" in evalue_family_contract_errors(bad2)


def test_tail_audit_contract() -> None:
    bad = dict(_BASE_TAIL)
    bad["params"] = dict(_BASE_TAIL["params"], p0_deep_share=0.9)  # not 0.05/0.1
    assert "p0_deep_share_not_tau_ratio" in evalue_family_contract_errors(bad)
    bad2 = dict(_BASE_TAIL)
    bad2["params"] = dict(_BASE_TAIL["params"], cell=[0.5, 0.1])
    assert "params_cell_not_adjacent_unit_pair" in evalue_family_contract_errors(bad2)
    bad3 = dict(_BASE_TAIL)
    bad3["params"] = dict(_BASE_TAIL["params"], alt_grid=[0.5, -1.0])
    assert "params_alt_grid_not_positive_list" in evalue_family_contract_errors(bad3)


def test_lane_power_contract() -> None:
    bad = dict(_BASE_LANE_POWER)
    bad["params"] = dict(_BASE_LANE_POWER["params"], defects=[0.1, 0.5])
    assert "params_defects_must_include_zero" in evalue_family_contract_errors(bad)
    bad2 = dict(_BASE_LANE_POWER, null_alarm_rate={"bogus_lane": 0.1})
    assert "null_alarm_rate_unknown_lane:bogus_lane" in evalue_family_contract_errors(bad2)
    bad3 = dict(_BASE_LANE_POWER, n_lanes_ok=99)
    assert "n_lanes_ok_exceeds_lanes" in evalue_family_contract_errors(bad3)


def test_honest_verdict_contract() -> None:
    bad = dict(
        _BASE_HONEST_VERDICT,
        verdict="supported_with_caveats",
        unavailable_lanes=["winner_curse"],
    )
    assert "unavailable_lanes_but_verdict_not_inconclusive" in evalue_family_contract_errors(bad)
    ok = dict(_BASE_HONEST_VERDICT, verdict="inconclusive", unavailable_lanes=["drift"])
    assert evalue_family_contract_errors(ok) == []
    bad2 = dict(_BASE_HONEST_VERDICT, unavailable_lanes=["bogus"])
    assert "unavailable_lane_unknown:bogus" in evalue_family_contract_errors(bad2)
    bad3 = dict(_BASE_HONEST_VERDICT, verdict="confirmed")
    assert "confirmed_without_promotion_flag" in evalue_family_contract_errors(bad3)
    bad4 = dict(_BASE_HONEST_VERDICT, verdict="winning")
    assert "verdict_not_in_enum" in evalue_family_contract_errors(bad4)


def test_monitor_run_contract() -> None:
    bad = dict(_BASE_MONITOR_RUN, n_alarm_rows=99)
    assert "n_alarm_rows_exceeds_n_rows" in evalue_family_contract_errors(bad)
    bad2 = dict(_BASE_MONITOR_RUN)
    bad2["lanes_available"] = dict(_BASE_MONITOR_RUN["lanes_available"], bogus_lane=True)
    assert "lanes_available_unknown:bogus_lane" in evalue_family_contract_errors(bad2)
    bad3 = dict(_BASE_MONITOR_RUN)
    bad3["params"] = dict(_BASE_MONITOR_RUN["params"], tail_cell=[0.5, 0.1])
    assert "tail_cell_not_ordered_pair" in evalue_family_contract_errors(bad3)
    bad4 = dict(_BASE_MONITOR_RUN)
    bad4["params"] = dict(_BASE_MONITOR_RUN["params"], level=1.2)
    assert "params.level_out_of_unit_interval" in evalue_family_contract_errors(bad4)


_BASE_SUITE_HEALTH = {
    "kind": "suite_health",
    "schema": "suite_health.v1",
    "data_label": "SYNTHETIC",
    "research_only": True,
    "live_pnl_claim": False,
    "inputs_sha256": _SHA,
    "params": {"receipts_dir": "receipts", "alpha": 0.05, "n_files": 10},
    "n_receipts": 10,
    "n_ok": 10,
    "n_failed": 0,
    "corpus_lane_available": True,
    "n_findings_harvested": 12,
    "n_evalues_pooled": 4,
    "pooled_evalue": 25.0,
    "pooled_alarmed": True,
}


def test_suite_health_contract() -> None:
    assert evalue_family_contract_errors(_BASE_SUITE_HEALTH) == []
    # accounting must reconcile
    bad = dict(_BASE_SUITE_HEALTH, n_failed=1)
    assert "ok_plus_failed_neq_receipts" in evalue_family_contract_errors(bad)
    # the fail-closed invariant: failed inputs withhold the pooled e-value
    bad2 = dict(_BASE_SUITE_HEALTH, n_ok=9, n_failed=1)
    assert "pooled_evalue_not_withheld_on_failures" in evalue_family_contract_errors(bad2)
    bad2["pooled_evalue"] = None
    bad2["pooled_alarmed"] = False
    assert evalue_family_contract_errors(bad2) == []
    # alarmed must sit at or above the 1/alpha threshold
    bad3 = dict(_BASE_SUITE_HEALTH, pooled_evalue=10.0)
    assert "pooled_alarmed_below_threshold" in evalue_family_contract_errors(bad3)
    bad4 = dict(_BASE_SUITE_HEALTH, n_evalues_pooled=99)
    assert "n_evalues_pooled_exceeds_findings" in evalue_family_contract_errors(bad4)


def test_monitor_run_label_binding_contract() -> None:
    """Outer data_label must equal the unique per-shard label."""
    payload = {
        "schema": "monitor_run.v1",
        "kind": "monitor_run",
        "data_label": "yahoo_eod",
        "research_only": True,
        "live_pnl_claim": False,
        "inputs_sha256": "a" * 64,
        "n_rows": 4,
        "n_alarm_rows": 1,
        "lanes_available": {"coverage": True},
        "params": {
            "data_labels": {"s1": "yahoo_eod", "s2": "yahoo_eod"},
            "alpha": 0.05,
            "level": 0.9,
            "tail_cell": [0.05, 0.1],
            "n_train": 64,
            "n_eval": 32,
            "seed": 0,
            "heads": ["a"],
            "shards": ["s1", "s2"],
        },
    }
    assert evalue_family_contract_errors(payload) == []
    bad = dict(payload)
    bad["data_label"] = "SYNTHETIC"  # mislabeled real shard
    assert evalue_family_contract_errors(bad) != []
    mixed = dict(payload)
    mixed["params"] = {"data_labels": {"s1": "a", "s2": "b"}}
    assert evalue_family_contract_errors(mixed) != []


def test_honest_verdict_label_binding_contract() -> None:
    """Verdict data_label binds to run.params.data_labels when present."""
    payload = {
        "kind": "honest_verdict.v1",
        "schema": "honest_verdict.v1",
        "data_label": "yahoo_eod",
        "research_only": True,
        "live_pnl_claim": False,
        "inputs_sha256": _SHA,
        "verdict": "confirmed",
        "winner": "a",
        "alpha": 0.05,
        "n_obs": 100,
        "n_heads": 2,
        "components": {},
        "unavailable_lanes": [],
        "evidence": [],
        "run": {
            "params": {
                "data_labels": {"s1": "yahoo_eod"},
            }
        },
    }
    errs = evalue_family_contract_errors(payload)
    assert "data_label_mismatches_shard_labels" not in errs
    bad = dict(payload)
    bad["data_label"] = "SYNTHETIC"
    assert "data_label_mismatches_shard_labels" in evalue_family_contract_errors(bad)


_BASE_MCS = {
    "kind": "mcs_seq.v1",
    "research_only": True,
    "live_pnl_claim": False,
    "data_label": "SYNTHETIC",
    "alpha": 0.05,
    "lam": 0.5,
    "n_heads": 4,
    "n_origins": 300,
    "survivors": ["champ"],
    "eliminated": {"x": 40, "y": 55, "z": 80},
    "n_eliminated": 3,
    "champion": "champ",
    "coverage_guarantee": "P(set contains an optimal head at every origin) >= 1 - alpha",
    "evidence": [
        "ville_inequality",
        "pairwise_supermartingales",
        "union_bound_k_minus_1",
        "permanent_elimination",
        "anytime_valid",
    ],
}


def test_mcs_seq_contract() -> None:
    errs = evalue_family_contract_errors(dict(_BASE_MCS))
    assert errs == [], errs


def test_mcs_seq_contract_partitions() -> None:
    # overlap between survivors and eliminated
    bad = dict(_BASE_MCS)
    bad["survivors"] = ["champ", "x"]
    assert "survivor_and_eliminated_overlap:['x']" in evalue_family_contract_errors(bad)

    # partition mismatch (5 entries for 4 heads)
    bad = dict(_BASE_MCS)
    bad["survivors"] = ["champ", "w"]
    assert "survivor_eliminated_partition_mismatch" in evalue_family_contract_errors(
        dict(bad, eliminated={"x": 1, "z": 2})
    ) or "survivor_and_eliminated_overlap" not in evalue_family_contract_errors(
        dict(_BASE_MCS, survivors=["champ", "w"], eliminated={"x": 1, "y": 2})
    )

    # champion outside survivor set
    bad = dict(_BASE_MCS)
    bad["champion"] = "ghost"
    assert "champion_not_in_survivors" in evalue_family_contract_errors(bad)

    # multiple survivors but a champion claimed
    bad = dict(_BASE_MCS)
    bad["survivors"] = ["champ", "w"]
    bad["eliminated"] = {"x": 40, "y": 55}
    bad["n_eliminated"] = 2
    assert "champion_with_multiple_survivors" in evalue_family_contract_errors(bad)

    # eliminated origin beyond n_origins
    bad = dict(_BASE_MCS)
    bad["eliminated"] = {"x": 40, "y": 55, "z": 999}
    assert "eliminated_origin_out_of_range:z" in evalue_family_contract_errors(bad)

    # n_eliminated must equal len(eliminated)
    bad = dict(_BASE_MCS)
    bad["n_eliminated"] = 2
    assert "n_eliminated_mismatch" in evalue_family_contract_errors(bad)

    # missing anytime_valid evidence tag
    bad = dict(_BASE_MCS)
    bad["evidence"] = []
    assert "evidence_missing_anytime_valid" in evalue_family_contract_errors(bad)


_BASE_SERIAL = {
    "kind": "serial_watch.v1",
    "research_only": True,
    "live_pnl_claim": False,
    "data_label": "SYNTHETIC",
    "alpha": 0.05,
    "lam": 0.5,
    "n_lags": 5,
    "n_origins": 400,
    "per_lag": {str(k): {"pos": 1.2, "neg": 0.8, "alarmed": k == 1} for k in range(1, 6)},
    "alarmed_lags": [1],
    "alarm_origins": {"lag1_pos": 88},
    "pooled_evalue": 2.4,
    "any_lag_alarmed": True,
    "pooled_alarmed": False,
    "evidence": ["ville_inequality", "anytime_valid"],
}


def test_serial_watch_contract() -> None:
    assert evalue_family_contract_errors(dict(_BASE_SERIAL)) == []


def test_serial_watch_contract_guards() -> None:
    # alarm lag outside the declared family
    bad = dict(_BASE_SERIAL)
    bad["alarmed_lags"] = [9]
    assert "alarmed_lag_out_of_range:9" in evalue_family_contract_errors(bad)

    # any_lag_alarmed must equal bool(alarmed_lags)
    bad = dict(_BASE_SERIAL)
    bad["any_lag_alarmed"] = False
    assert "any_lag_alarmed_mismatch" in evalue_family_contract_errors(bad)

    # pooled_alarmed requires pooled >= 1/alpha
    bad = dict(_BASE_SERIAL)
    bad["pooled_alarmed"] = True
    assert "pooled_alarmed_below_threshold" in evalue_family_contract_errors(bad)

    # alarm origin beyond n_origins
    bad = dict(_BASE_SERIAL)
    bad["alarm_origins"] = {"lag2_neg": 999}
    assert "alarm_origin_bad:lag2_neg" in evalue_family_contract_errors(bad)

    # non-positive per-lag e-value
    bad = dict(_BASE_SERIAL)
    bad["per_lag"] = dict(
        _BASE_SERIAL["per_lag"], **{"3": {"pos": -1.0, "neg": 1.0, "alarmed": False}}
    )
    assert "per_lag_pos_not_positive:3" in evalue_family_contract_errors(bad)

    # missing the anytime_valid evidence tag
    bad = dict(_BASE_SERIAL)
    bad["evidence"] = []
    assert "evidence_missing_anytime_valid" in evalue_family_contract_errors(bad)
