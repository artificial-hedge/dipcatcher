"""Adversarial probes for the bench/cli research lane.

Seeded SYNTHETIC probes only. Covers the audit-lane fixes:

- ``identity_sweep._r_kyle_lambda_ofi_depth_corr``: an undefined (NaN)
  correlation or HAC p-value on an adequately-aligned panel used to slip
  past the bound check as a silent pass; it must now fail closed.
- ``crossvenue_basis.crossvenue_basis_contract_errors``: a non-object
  ``inputs.legs`` entry used to slip past the label-derivation and
  dataset-digest loops; it must now report ``inputs_legs_meta_not_object``.

Plus fail-closed contract probes on the audited surface (vol_bench,
stocbench allocation/significance) that pinned boundaries must keep.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from quant_fund.research import crossvenue_basis, identity_sweep, stocbench, vol_bench


def test_kyle_corr_nan_spearman_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        identity_sweep,
        "kyle_lambda_ofi_depth_corr",
        lambda fused, min_names=3: {
            "n_dates_aligned": 5,
            "kyle_lambda_ofi_depth_spearman": math.nan,
            "kyle_lambda_ofi_depth_pearson": 0.4,
            "kyle_lambda_ofi_depth_prod_hac_p": 0.2,
        },
    )
    bundle = types.SimpleNamespace(fused=None)
    with pytest.raises(ValueError, match="spearman undefined"):
        identity_sweep._r_kyle_lambda_ofi_depth_corr(bundle)


def test_kyle_corr_nan_pearson_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        identity_sweep,
        "kyle_lambda_ofi_depth_corr",
        lambda fused, min_names=3: {
            "n_dates_aligned": 4,
            "kyle_lambda_ofi_depth_spearman": -0.1,
            "kyle_lambda_ofi_depth_pearson": math.inf,
            "kyle_lambda_ofi_depth_prod_hac_p": 0.9,
        },
    )
    bundle = types.SimpleNamespace(fused=None)
    with pytest.raises(ValueError, match="pearson undefined"):
        identity_sweep._r_kyle_lambda_ofi_depth_corr(bundle)


def test_kyle_corr_nan_hac_p_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        identity_sweep,
        "kyle_lambda_ofi_depth_corr",
        lambda fused, min_names=3: {
            "n_dates_aligned": 9,
            "kyle_lambda_ofi_depth_spearman": 0.0,
            "kyle_lambda_ofi_depth_pearson": 0.0,
            "kyle_lambda_ofi_depth_prod_hac_p": math.nan,
        },
    )
    bundle = types.SimpleNamespace(fused=None)
    with pytest.raises(ValueError, match="prod_hac_p undefined"):
        identity_sweep._r_kyle_lambda_ofi_depth_corr(bundle)


def test_kyle_corr_short_alignment_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        identity_sweep,
        "kyle_lambda_ofi_depth_corr",
        lambda fused, min_names=3: {
            "n_dates_aligned": 2,
            "kyle_lambda_ofi_depth_spearman": math.nan,
            "kyle_lambda_ofi_depth_pearson": math.nan,
            "kyle_lambda_ofi_depth_prod_hac_p": math.nan,
        },
    )
    bundle = types.SimpleNamespace(fused=None)
    with pytest.raises(ValueError, match=">= 3 aligned dates"):
        identity_sweep._r_kyle_lambda_ofi_depth_corr(bundle)


def test_kyle_corr_out_of_range_reports_bound_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        identity_sweep,
        "kyle_lambda_ofi_depth_corr",
        lambda fused, min_names=3: {
            "n_dates_aligned": 6,
            "kyle_lambda_ofi_depth_spearman": 1.5,
            "kyle_lambda_ofi_depth_pearson": -0.2,
            "kyle_lambda_ofi_depth_prod_hac_p": 1.4,
        },
    )
    bundle = types.SimpleNamespace(fused=None)
    worst = identity_sweep._r_kyle_lambda_ofi_depth_corr(bundle)
    assert worst == pytest.approx(0.5)


def _crossvenue_receipt(legs_meta: object) -> dict:
    params = {"tol": 0.001}
    return {
        "schema": crossvenue_basis.CROSSVENUE_SCHEMA,
        "kind": crossvenue_basis.CROSSVENUE_KIND,
        "research_only": True,
        "live_pnl_claim": False,
        "simulated_only": False,
        "data_label": "SYNTHETIC",
        "verdict": "measured",
        "inputs": {"params": params, "legs": legs_meta},
        "params": params,
    }


def test_crossvenue_rejects_non_object_leg_meta() -> None:
    receipt = _crossvenue_receipt({"kraken": 42})
    errors = crossvenue_basis.crossvenue_basis_contract_errors(receipt)
    assert "inputs_legs_meta_not_object" in errors


def test_crossvenue_rejects_string_leg_meta() -> None:
    receipt = _crossvenue_receipt({"kraken": "SYNTHETIC"})
    errors = crossvenue_basis.crossvenue_basis_contract_errors(receipt)
    assert "inputs_legs_meta_not_object" in errors


def test_vol_bench_contract_flags_forbidden_metric_key() -> None:
    receipt = {
        "schema": vol_bench.VOL_BENCH_SCHEMA,
        "kind": "vol_bench",
        "data_label": "SYNTHETIC",
        "live_pnl_claim": False,
        "sharpe": 1.7,
        "results": [{"status": "ok"}],
        "n_rows": 1,
        "n_error_rows": 0,
    }
    errors = vol_bench.vol_bench_contract_errors(receipt)
    assert "forbidden_metric_keys" in errors


def test_vol_bench_contract_flags_lying_counts() -> None:
    receipt = {
        "schema": vol_bench.VOL_BENCH_SCHEMA,
        "kind": "vol_bench",
        "data_label": "SYNTHETIC",
        "live_pnl_claim": False,
        "results": [{"status": "ok"}, {"status": "error"}],
        "n_rows": 1,
        "n_error_rows": 0,
    }
    errors = vol_bench.vol_bench_contract_errors(receipt)
    assert "n_rows_mismatch" in errors
    assert "n_error_rows_mismatch" in errors


def test_allocate_budget_floors_and_sums() -> None:
    alloc = stocbench.allocate_budget(23, 5, 2, rule="equal")
    assert alloc.sum() == 23 // 2
    assert bool((alloc >= 2).all())
    with pytest.raises(ValueError, match="cannot cover"):
        stocbench.allocate_budget(9, 5, 1, rule="equal")
    with pytest.raises(ValueError, match="pilot_variance is required"):
        stocbench.allocate_budget(40, 5, 1, rule="neyman")
    with pytest.raises(ValueError, match="nonnegative"):
        stocbench.allocate_budget(40, 5, 1, rule="neyman", pilot_variance=np.array([-1.0] * 5))


def test_neyman_allocation_respects_floor_and_budget() -> None:
    pilot = np.array([0.0, 1.0, 4.0, 0.25, 9.0])
    alloc = stocbench.allocate_budget(100, 5, 2, rule="neyman", pilot_variance=pilot)
    assert alloc.sum() == 50
    assert bool((alloc >= 2).all())
    assert alloc[4] >= alloc[1]  # higher pilot variance earns more draws


def test_budget_significance_on_decisive_stream() -> None:
    rng = np.random.default_rng(7)
    diffs = 1.0 + 0.2 * rng.standard_normal(40)
    sig = stocbench.budget_significance(diffs, sigma=0.5, alpha=0.05)
    assert sig.significant_within_budget
    assert sig.first_excluding > 0
    assert sig.final_lower > 0.0


def test_budget_significance_rejects_bad_sigma() -> None:
    with pytest.raises(ValueError, match="sigma"):
        stocbench.budget_significance(np.array([0.1, 0.2, 0.1]), sigma=0.0)
    with pytest.raises(ValueError, match="sigma"):
        stocbench.budget_significance(np.array([0.1, 0.2, 0.1]), sigma=math.nan)


def test_ensemble_energy_distance_zero_iff_identical() -> None:
    rng = np.random.default_rng(11)
    x = rng.standard_normal((64, 2))
    assert stocbench.ensemble_energy_distance(x, x.copy()) == pytest.approx(0.0)
    shifted = x + 1.0
    assert stocbench.ensemble_energy_distance(x, shifted) > 0.0
