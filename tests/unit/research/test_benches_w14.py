"""Tests for research/benches_w14.py — SOTA canon wave 14 scorecard families.

Each bench is executed once per module run (module-scoped fixtures): the values
are deterministic (seeded from a module constant; the torch trainers pin
``torch.manual_seed`` and run CPU single-thread), so re-running a bench per test
buys nothing. The determinism tests at the bottom re-run each bench once and
compare against its fixture bit-for-bit.

Tolerance policy: the wave-14 benches run SHRUNK Monte-Carlo / training budgets
relative to the lane suites in tests/unit/core, tests/unit/models and
tests/unit/microstructure (e.g. 12k/6k IS draws instead of 25k/8k, an n_t<=50
MFG grid instead of up to 400, a T=100 GAS series instead of 1500, 20k/100-step
Malliavin MC instead of 30k/126, 8k/2-iter SLV instead of 12k/3-iter, 200-epoch
deep BSDE instead of 500, 120-epoch DeRegiME instead of 400, 1500s ZI-LOB
sessions instead of 3000s). The science assertions below are therefore
DIRECTIONAL with wider documented slack than the lane tests — they pin the
qualitative claims of the cited papers (model-free bound containment, IS
variance reduction, the crowding externality, square-root impact, sell-first
execution, hedged < unhedged risk, DeRegiME/TORF proper-score gains) without
asserting the lane suites' tight MC windows.

Four families (deep_bsde, deep_regime_mixture, odd_residual_flows,
deep_kernel_hedging) are torch-gated: their science tests skip when the optional
``nn`` extra is absent, in which case the benches legitimately return ``{}``.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from quant_fund.research.benches_w14 import (
    bench_american_lsm,
    bench_cash_constrained_oe,
    bench_deep_bsde,
    bench_deep_kernel_hedging,
    bench_deep_regime_mixture,
    bench_large_deviations,
    bench_local_stoch_vol,
    bench_malliavin_greeks,
    bench_martingale_ot,
    bench_mean_field_games,
    bench_odd_residual_flows,
    bench_vine_copula,
    bench_xva,
    bench_zi_lob,
)
from quant_fund.research.catalog import (
    OPTIONAL_BENCHMARK_FAMILIES,
    family_blob_forbidden_metrics_absent,
    family_blob_has_finite_observation,
)


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="wave-14 deep-learning benches require the nn extra (torch)"
)

_FAMILIES = (
    "martingale_ot",
    "large_deviations",
    "mean_field_games",
    "vine_copula",
    "malliavin_greeks",
    "xva",
    "american_lsm",
    "local_stoch_vol",
    "deep_bsde",
    "deep_regime_mixture",
    "odd_residual_flows",
    "deep_kernel_hedging",
    "zi_lob",
    "cash_constrained_oe",
)
# The ten numpy/scipy families (the four deep-learning ones are torch-gated).
_NUMPY_BLOBS = (
    "martingale_ot",
    "large_deviations",
    "mean_field_games",
    "vine_copula",
    "malliavin_greeks",
    "xva",
    "american_lsm",
    "local_stoch_vol",
    "zi_lob",
    "cash_constrained_oe",
)
_TORCH_BLOBS = (
    "deep_bsde",
    "deep_regime_mixture",
    "odd_residual_flows",
    "deep_kernel_hedging",
)


@pytest.fixture(scope="module")
def martingale_ot() -> dict[str, float]:
    return bench_martingale_ot()


@pytest.fixture(scope="module")
def large_deviations() -> dict[str, float]:
    return bench_large_deviations()


@pytest.fixture(scope="module")
def mean_field_games() -> dict[str, float]:
    return bench_mean_field_games()


@pytest.fixture(scope="module")
def vine_copula() -> dict[str, float]:
    return bench_vine_copula()


@pytest.fixture(scope="module")
def malliavin_greeks() -> dict[str, float]:
    return bench_malliavin_greeks()


@pytest.fixture(scope="module")
def xva() -> dict[str, float]:
    return bench_xva()


@pytest.fixture(scope="module")
def american_lsm() -> dict[str, float]:
    return bench_american_lsm()


@pytest.fixture(scope="module")
def local_stoch_vol() -> dict[str, float]:
    return bench_local_stoch_vol()


@pytest.fixture(scope="module")
def deep_bsde() -> dict[str, float]:
    return bench_deep_bsde()


@pytest.fixture(scope="module")
def deep_regime_mixture() -> dict[str, float]:
    return bench_deep_regime_mixture()


@pytest.fixture(scope="module")
def odd_residual_flows() -> dict[str, float]:
    return bench_odd_residual_flows()


@pytest.fixture(scope="module")
def deep_kernel_hedging() -> dict[str, float]:
    return bench_deep_kernel_hedging()


@pytest.fixture(scope="module")
def zi_lob() -> dict[str, float]:
    return bench_zi_lob()


@pytest.fixture(scope="module")
def cash_constrained_oe() -> dict[str, float]:
    return bench_cash_constrained_oe()


def test_families_registered_as_optional() -> None:
    for fam in _FAMILIES:
        assert fam in OPTIONAL_BENCHMARK_FAMILIES


@pytest.mark.parametrize("blob_name", list(_NUMPY_BLOBS))
def test_numpy_blobs_are_clean_and_finite(blob_name: str, request: pytest.FixtureRequest) -> None:
    blob = request.getfixturevalue(blob_name)
    assert isinstance(blob, dict) and blob
    assert family_blob_has_finite_observation(blob)
    assert family_blob_forbidden_metrics_absent(blob)
    assert all(np.isfinite(v) for v in blob.values())


@pytest.mark.parametrize("blob_name", list(_TORCH_BLOBS))
def test_torch_blobs_are_clean(blob_name: str, request: pytest.FixtureRequest) -> None:
    blob = request.getfixturevalue(blob_name)
    # Forbidden-key scan holds whether or not torch is present.
    assert family_blob_forbidden_metrics_absent(blob)
    if not _HAS_TORCH:
        assert blob == {}
        return
    assert isinstance(blob, dict) and blob
    assert family_blob_has_finite_observation(blob)
    assert all(np.isfinite(v) for v in blob.values())


def test_martingale_ot_bounds_and_duality(martingale_ot: dict[str, float]) -> None:
    # BHP 2013 Thm 1: the LP has no duality gap (exact HiGHS solve).
    assert martingale_ot["mot_upper_bound"] >= martingale_ot["mot_lower_bound"]
    assert martingale_ot["mot_spread"] > 0.0
    assert martingale_ot["mot_upper_dual_gap"] < 1e-6
    # The known BS exotic prices land inside the model-free [lower, upper] band:
    # the variance-swap fair strike sigma^2 (bs_contained) and a seeded butterfly.
    assert martingale_ot["mot_bs_contained"] == 1.0
    assert martingale_ot["mot_butterfly_lower"] <= martingale_ot["mot_butterfly_upper"]


def test_large_deviations_is_and_chernoff(large_deviations: dict[str, float]) -> None:
    # Glasserman-Li exponential tilting beats crude MC on a ~2-sigma tail.
    assert large_deviations["ld_is_variance_reduction_factor"] > 2.0
    # Cramer-Chernoff bound dominates the empirical tail (with MC slack).
    assert large_deviations["ld_chernoff_bound_holds"] == 1.0
    # Numerical Fenchel-Legendre reproduces the closed-form Gaussian rate.
    assert large_deviations["ld_gaussian_rate_closed_form_err"] < 1e-6


def test_mean_field_games_ac_crowding_and_grid(mean_field_games: dict[str, float]) -> None:
    # gamma=0 hard-liquidation LQ equilibrium recovers the Almgren-Chriss rate
    # (lane asserts mse < 0.01 at n_t=400; the shrunk n_t=50 grid widens it).
    assert mean_field_games["mfg_ac_recovery_mse"] < 0.02
    # Crowding externality: positive coupling lifts equilibrium cost above solitary.
    assert mean_field_games["mfg_crowding_cost_ratio"] > 1.0
    # Grid convergence: coarse-minus-fine discretisation gap is positive.
    assert mean_field_games["mfg_grid_convergence"] > 0.0


def test_vine_copula_recovery_tail_and_gas(vine_copula: dict[str, float]) -> None:
    # C-vine Gaussian structure recovery of the planted tau=0.4 (lane asserts
    # |rho_hat - rho_true| < 0.15 at n=3000; the shrunk n=2000 tau error tracks it).
    assert vine_copula["vine_structure_recovery_tau_err"] < 0.15
    # Clayton lower-tail dependence strictly dominates the Gaussian copula's.
    assert vine_copula["vine_tail_dep_ratio"] > 1.0
    # GAS(1,1) parameter recovery bias (lane tolerances 0.25/0.25/0.30).
    assert vine_copula["gas_recov_bias_max"] < 0.30


def test_malliavin_greeks_accuracy_and_efficiency(malliavin_greeks: dict[str, float]) -> None:
    # Malliavin Delta matches the closed form within a few standard errors.
    assert malliavin_greeks["mall_delta_abs_err_se"] < 3.0
    # Gamma is a second-order weight and carries the discretisation bias of the
    # shrunk 100-step Euler scheme (lane allows 3*SE + 0.01 absolute slack);
    # the diagnostic stays finite and O(a few SE).
    assert malliavin_greeks["mall_gamma_abs_err_se"] < 12.0
    # Fournie et al. 1999 headline: on a digital payoff the Malliavin Delta
    # variance is far below the finite-difference Delta variance.
    assert malliavin_greeks["mall_digital_fd_efficiency_ratio"] > 1.0


def test_xva_levels_and_wwr(xva: dict[str, float]) -> None:
    assert xva["xva_cva"] > 0.0
    assert xva["xva_fva"] >= 0.0
    assert xva["xva_mva"] >= 0.0
    # Gaussian-copula wrong-way risk raises CVA above the rho=0 baseline.
    assert xva["xva_wwr_uplift"] > 0.0
    # Closed-form deterministic-exposure CVA anchor is reproduced exactly.
    assert xva["xva_closed_form_abs_err"] < 1e-8


def test_american_lsm_dual_bracket(american_lsm: dict[str, float]) -> None:
    # Andersen-Broadie duality: the LSM lower bound sits under the dual upper.
    assert american_lsm["lsm_oos_lower_price"] <= american_lsm["lsm_dual_upper_bound"]
    assert american_lsm["lsm_duality_gap"] >= 0.0
    assert american_lsm["lsm_lower_le_upper"] == 1.0
    # The LSM price agrees with the Barone-Adesi-Whaley reference within MC error.
    assert american_lsm["lsm_baw_agreement_ok"] == 1.0


def test_local_stoch_vol_dupire_and_slv(local_stoch_vol: dict[str, float]) -> None:
    # Fundamental Dupire consistency: LV-MC implied vols reproduce the input
    # SABR surface to < 2 vol points inside the central moneyness band.
    assert local_stoch_vol["slv_dupire_roundtrip_max_volpts"] < 2.0
    # Matched-Heston SLV calibration lands on the CF target to < 1 vol point.
    assert local_stoch_vol["slv_final_iv_max_abs_volpts"] < 1.0
    # Zero vol-of-vol degenerate limit: the leverage fixed point is exactly 1.
    assert local_stoch_vol["slv_xi0_leverage_deviation"] < 1e-9


def test_zi_lob_impact_and_regime(zi_lob: dict[str, float]) -> None:
    # Donier et al. 2015 square-root law: the emergent log-log impact slope is
    # near 0.5 with a clean fit.
    assert 0.40 <= zi_lob["zlob_impact_slope"] <= 0.60
    assert zi_lob["zlob_impact_r2"] >= 0.80
    # Moret & Lillo 2026 motivation: a stationarily-calibrated market maker
    # saturates its inventory under persistent directional flow (>> stationary).
    assert zi_lob["zlob_inventory_saturation_ratio"] > 2.0
    assert zi_lob["zlob_regime_detected"] == 1.0


def test_cash_constrained_oe_sell_first_and_drawdown(cash_constrained_oe: dict[str, float]) -> None:
    # Hashimoto & Stillman 2026: a tighter cash budget cuts the peak funding
    # drawdown and shifts the schedule toward sell-first execution, at only a
    # modest implementation-shortfall cost (nested feasible sets).
    assert cash_constrained_oe["coe_peak_drawdown_reduction"] > 1.0
    assert (
        cash_constrained_oe["coe_sell_first_fraction_tight"]
        > cash_constrained_oe["coe_sell_first_fraction_unconstrained"]
    )
    assert cash_constrained_oe["coe_is_ratio_tight_vs_unconstrained"] >= 1.0


@requires_torch
def test_deep_bsde_recovers_burgers_hopf(deep_bsde: dict[str, float]) -> None:
    # E-Han-Jentzen Lemma 4.3: the recovered Y_0 approaches u(0,0)=1/2.
    assert deep_bsde["bsde_burgers_d1_abs_err"] < 0.02
    assert deep_bsde["bsde_terminal_loss"] >= 0.0


@requires_torch
def test_deep_regime_mixture_beats_ngboost(deep_regime_mixture: dict[str, float]) -> None:
    # DeRegiME's regime-mixture Student-t head beats the Gaussian NGBoost NLPD.
    assert deep_regime_mixture["drm_nlpd_gap_vs_ngboost"] < 0.0
    assert deep_regime_mixture["drm_planted_regimes"] == 3.0
    assert deep_regime_mixture["drm_effective_regimes"] >= 1.0


@requires_torch
def test_odd_residual_flows_crps_and_mae(odd_residual_flows: dict[str, float]) -> None:
    # TORF matches/beats the NGBoost Gaussian CRPS on the heteroskedastic stream.
    assert odd_residual_flows["crps_gain_vs_ngboost"] > 0.0
    assert odd_residual_flows["torf_crps"] > 0.0
    assert odd_residual_flows["ngboost_crps"] > 0.0
    # Lemma 1: the Stage-1 point forecast is preserved EXACTLY (bitwise 0.0).
    assert odd_residual_flows["mae_preservation_gap"] == 0.0


@requires_torch
def test_deep_kernel_hedging_reduces_risk(deep_kernel_hedging: dict[str, float]) -> None:
    # Dupret et al. 2026: the RKHS hedge beats the unhedged payoff variance.
    assert deep_kernel_hedging["dkh_hedged_risk"] < deep_kernel_hedging["dkh_unhedged_risk"]
    assert deep_kernel_hedging["dkh_rff_kernel_max_error"] >= 0.0


def test_numpy_benches_are_deterministic(
    martingale_ot: dict[str, float],
    large_deviations: dict[str, float],
    mean_field_games: dict[str, float],
    vine_copula: dict[str, float],
    malliavin_greeks: dict[str, float],
    xva: dict[str, float],
    american_lsm: dict[str, float],
    local_stoch_vol: dict[str, float],
    zi_lob: dict[str, float],
    cash_constrained_oe: dict[str, float],
) -> None:
    # Seeded from module constants (and cvxpy/CLARABEL is deterministic), so a
    # fresh call must reproduce each fixture bit-for-bit.
    assert bench_martingale_ot() == martingale_ot
    assert bench_large_deviations() == large_deviations
    assert bench_mean_field_games() == mean_field_games
    assert bench_vine_copula() == vine_copula
    assert bench_malliavin_greeks() == malliavin_greeks
    assert bench_xva() == xva
    assert bench_american_lsm() == american_lsm
    assert bench_local_stoch_vol() == local_stoch_vol
    assert bench_zi_lob() == zi_lob
    assert bench_cash_constrained_oe() == cash_constrained_oe


@requires_torch
def test_torch_benches_are_deterministic(
    deep_bsde: dict[str, float],
    deep_regime_mixture: dict[str, float],
    odd_residual_flows: dict[str, float],
    deep_kernel_hedging: dict[str, float],
) -> None:
    # The torch trainers pin torch.manual_seed and run CPU single-thread, so a
    # fresh call must reproduce each fixture bit-for-bit.
    assert bench_deep_bsde() == deep_bsde
    assert bench_deep_regime_mixture() == deep_regime_mixture
    assert bench_odd_residual_flows() == odd_residual_flows
    assert bench_deep_kernel_hedging() == deep_kernel_hedging
