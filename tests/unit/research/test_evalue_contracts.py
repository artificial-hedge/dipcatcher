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
    bad = dict(_BASE_PROMOTION, anytime_p=0.5)  # not 1/25
    assert "anytime_p_not_reciprocal_of_evalue" in evalue_family_contract_errors(bad)
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
    assert result["valid"] is False
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
