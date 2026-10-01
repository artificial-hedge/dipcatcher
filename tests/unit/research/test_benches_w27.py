"""Wave-27 canon wiring: OPTIONAL family registration + bench determinism.

Every family in this wave must appear in ``OPTIONAL_BENCHMARK_FAMILIES`` and
return a finite, forbidden-token-free blob when its lane module is present.
Science assertions stay permissive — they guard that the adapter returns the
family's real diagnostics, not that synthetic accuracy meets a bar.
"""

from __future__ import annotations

import math

import pytest

from quant_fund.research import benches_w27
from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

_FAMILIES_LANDED = (
    "durbin_koopman",
    "dp_mixture",
    "expert_aggregation",
    "instrumental_quantile",
    "implied_tree",
    "ensemble_kalman_inversion",
)


@pytest.fixture(scope="module")
def dk_blob() -> dict[str, float]:
    return benches_w27.bench_durbin_koopman()


@pytest.fixture(scope="module")
def dpmm_blob() -> dict[str, float]:
    return benches_w27.bench_dp_mixture()


@pytest.fixture(scope="module")
def expert_blob() -> dict[str, float]:
    return benches_w27.bench_expert_aggregation()


@pytest.fixture(scope="module")
def ivqr_blob() -> dict[str, float]:
    return benches_w27.bench_instrumental_quantile()


@pytest.fixture(scope="module")
def tree_blob() -> dict[str, float]:
    return benches_w27.bench_implied_tree()


@pytest.fixture(scope="module")
def eki_blob() -> dict[str, float]:
    return benches_w27.bench_ensemble_kalman_inversion()


def test_wave27_families_registered() -> None:
    for family in _FAMILIES_LANDED:
        assert family in OPTIONAL_BENCHMARK_FAMILIES


@pytest.mark.parametrize(
    ("adapter", "fixture_name"),
    [
        (benches_w27.bench_durbin_koopman, "dk_blob"),
        (benches_w27.bench_dp_mixture, "dpmm_blob"),
        (benches_w27.bench_expert_aggregation, "expert_blob"),
        (benches_w27.bench_instrumental_quantile, "ivqr_blob"),
        (benches_w27.bench_implied_tree, "tree_blob"),
        (benches_w27.bench_ensemble_kalman_inversion, "eki_blob"),
    ],
)
def test_family_blob_clean_contract(adapter, fixture_name, request) -> None:
    blob = request.getfixturevalue(fixture_name)
    assert blob, f"{adapter.__name__} returned empty blob"
    assert all(math.isfinite(v) for v in blob.values())
    forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
    for key in blob:
        assert forbidden.isdisjoint(key.lower().split("_"))


def test_durbin_koopman_blob_smooths(dk_blob) -> None:
    assert dk_blob["synthetic_smooth_corr"] > 0.9
    assert dk_blob["synthetic_smooth_ge_filter"] == 1.0
    assert dk_blob["synthetic_loglik_finite"] == 1.0
    assert 0.5 < dk_blob["synthetic_draw_cov_calib"] < 1.0


def test_dp_mixture_blob_recovers_regimes(dpmm_blob) -> None:
    assert dpmm_blob["synthetic_ari"] > 0.8
    assert dpmm_blob["synthetic_k_hat_err"] < 0.5
    assert dpmm_blob["synthetic_elbo_monotone_frac"] > 0.8
    assert dpmm_blob["synthetic_assign_sharpness"] > 0.9


def test_expert_aggregation_blob_tracks(expert_blob) -> None:
    assert expert_blob["synthetic_hedge_regret_ratio"] < 0.5
    assert expert_blob["synthetic_fixedshare_beats_static"] == 1.0
    assert expert_blob["synthetic_eta_bound_respected"] == 1.0
    assert expert_blob["synthetic_switch_detect_lag"] < 60


def test_instrumental_quantile_blob_identifies(ivqr_blob) -> None:
    assert abs(ivqr_blob["synthetic_ivqr_bias_median"]) < abs(
        ivqr_blob["synthetic_naive_qr_bias_median"]
    )
    assert ivqr_blob["synthetic_ar_size"] < 0.3
    assert ivqr_blob["synthetic_ar_power"] > 0.8
    assert ivqr_blob["synthetic_first_stage_f"] > 10
    assert ivqr_blob["synthetic_ci_covers_truth"] == 1.0


def test_implied_tree_blob_calibrates(tree_blob) -> None:
    assert tree_blob["synthetic_smile_fit_err"] < 0.1
    assert tree_blob["synthetic_density_mass_err"] < 1e-6
    assert tree_blob["synthetic_martingale_err"] < 1e-6
    assert tree_blob["synthetic_arb_violations"] == 0.0
    assert tree_blob["synthetic_path_claim_bounds"] == 1.0


def test_ensemble_kalman_inversion_blob_recovers(eki_blob) -> None:
    assert eki_blob["synthetic_param_relerr"] < 0.15
    assert eki_blob["synthetic_misfit_reduction"] > 5.0
    assert eki_blob["synthetic_spread_collapse"] < 0.2
    assert eki_blob["synthetic_monotone_misfit_frac"] > 0.7
    assert eki_blob["synthetic_determinism"] == 1.0
