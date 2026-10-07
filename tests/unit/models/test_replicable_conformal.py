"""Replicable conformal prediction (arXiv:2608.23638) — seeded SYNTHETIC correctness tests.

Papamichalis, Ruane & Papamichalis (2026), "Replicable Conformal Prediction".
All experiments run on the paper's rank-transformed (PIT-uniform) score
testbed: scores iid Unif[0, 1], so the score law is F(t) = t, the density is
f == 1 and kappa == 1 exactly, and a threshold tau has population coverage
min(tau, 1) — synthetic fixture, never market evidence. Metrics are proper
research scores (coverage, set size, agreement/identity rates); no
Sharpe/Sortino/Calmar/P&L/NAV anywhere (AGENTS.md honesty contract).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from quant_fund.metrics.conformal import conformal_quantile
from quant_fund.models.replicable_conformal import (
    _kth_order_stats,
    agreement_experiment,
    asymptotic_mismatch_limit,
    b_n_spread,
    bench_replicable_conformal,
    conditional_coverage_band,
    coverage_drop_approximation,
    disagreement_bound,
    epsilon_protocol_grid_width,
    expected_max_normal,
    expected_min_coverage,
    grid_width,
    pilot_density,
    protocol_sample_requirement,
    replicability_size_floor,
    replicable_conformal_threshold,
    round_up_to_grid,
    sample_lower_bound,
    seedless_experiment,
    seedless_sample_requirement,
    selective_recalibration_bound,
    selective_recalibration_experiment,
    shared_offset,
    validity_cost_experiment,
)

ALPHA = 0.10
RHO = 0.10


# ---------------------------------------------------------------------------
# Algorithm 1 core: grid rounding, shared seed, threshold construction
# ---------------------------------------------------------------------------


def test_round_up_to_grid_is_upward_and_grid_valued() -> None:
    beta, u = 0.1, 0.03
    v = np.array([-0.2, 0.0, 0.42, 0.8999, 1.5])
    tau = round_up_to_grid(v, beta, u)
    assert np.all(tau >= v - 1e-12)
    assert np.all(tau - v <= beta + 1e-12)
    cells = (tau - u) / beta
    assert np.allclose(cells, np.round(cells), atol=1e-9)
    # dyadic grid: exact arithmetic, grid points are fixed points of rounding
    b2, u2 = 0.25, 0.125
    v2 = np.array([0.0, 0.3125, 0.55, 0.875, 1.5])
    got = round_up_to_grid(v2, b2, u2)
    assert np.allclose(got, np.array([0.125, 0.375, 0.625, 0.875, 1.625]), atol=0.0, rtol=0.0)


def test_replicable_threshold_matches_algorithm_by_hand() -> None:
    rng = np.random.default_rng(7)
    scores = rng.random(100)
    alpha, beta, u = 0.10, 0.07, 0.02
    # Algorithm 1 step 1: k-th smallest score, k = ceil((1 - alpha)(n + 1)) = 91
    k = int(np.ceil((1.0 - alpha) * 101))
    assert k == 91
    tau_tilde = float(np.sort(scores)[k - 1])
    expected = u + beta * np.ceil((tau_tilde - u) / beta)
    got = replicable_conformal_threshold(scores, alpha, beta, u)
    assert got == pytest.approx(float(expected), abs=1e-12)
    assert got >= tau_tilde
    assert got - tau_tilde <= beta + 1e-12


def test_threshold_dominates_standard_quantile_over_seeds() -> None:
    for seed in range(50):
        rng = np.random.default_rng(seed)
        scores = rng.random(400)
        beta = 0.05
        u = shared_offset(1000 + seed, beta)
        tau = replicable_conformal_threshold(scores, ALPHA, beta, u)
        tau_std = conformal_quantile(scores, ALPHA)
        assert tau >= tau_std
        assert tau < tau_std + beta + 1e-12


def test_shared_seed_offset_bit_identical_and_in_range() -> None:
    beta = 0.125
    u1 = shared_offset(23638, beta)
    u2 = shared_offset(23638, beta)
    u3 = shared_offset(999, beta)
    assert u1 == u2  # bit-identical: same pre-registered seed
    assert u1 != u3
    assert 0.0 <= u1 < beta


def test_same_seed_same_scores_give_identical_threshold() -> None:
    rng = np.random.default_rng(3)
    scores = rng.random(500)
    beta = 0.04
    u = shared_offset(11, beta)
    t1 = replicable_conformal_threshold(scores, ALPHA, beta, u)
    t2 = replicable_conformal_threshold(scores.copy(), ALPHA, beta, u)
    assert t1 == t2


def test_proposition1_standard_split_conformal_never_replicates() -> None:
    """Prop 1: with continuous scores, independent standard thresholds never agree."""
    matches = 0
    n_pairs = 200
    base = np.random.SeedSequence(1234)
    for child in base.spawn(n_pairs):
        rng = np.random.default_rng(child)
        t_a = conformal_quantile(rng.random(2_000), ALPHA)
        t_b = conformal_quantile(rng.random(2_000), ALPHA)
        matches += int(t_a == t_b)
    assert matches == 0


def test_vectorized_kth_order_stats_matches_scalar() -> None:
    rng = np.random.default_rng(9)
    mat = rng.random((25, 300))
    got = _kth_order_stats(mat, ALPHA)
    for i in range(25):
        assert got[i] == pytest.approx(conformal_quantile(mat[i], ALPHA), abs=0.0)


# ---------------------------------------------------------------------------
# Fail-closed edges
# ---------------------------------------------------------------------------


def test_fail_closed_grid_edges() -> None:
    scores = np.array([0.5, 0.6, 0.7, 0.8])
    for beta in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="beta"):
            round_up_to_grid(scores, beta, 0.0)
        with pytest.raises(ValueError, match="beta"):
            replicable_conformal_threshold(scores, 0.3, beta, 0.0)
    with pytest.raises(ValueError, match="offset"):
        round_up_to_grid(scores, 0.1, -1e-12)
    with pytest.raises(ValueError, match="offset"):
        round_up_to_grid(scores, 0.1, 0.1)
    with pytest.raises(ValueError, match="beta"):
        shared_offset(1, 0.0)


def test_fail_closed_alpha_edges() -> None:
    scores = np.linspace(0.05, 0.95, 40)
    for alpha in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            replicable_conformal_threshold(scores, alpha, 0.1, 0.0)
        with pytest.raises(ValueError, match="alpha"):
            b_n_spread(alpha, 100)
        with pytest.raises(ValueError, match="alpha"):
            grid_width(alpha, 100, RHO, 1.0)


def test_fail_closed_unattainable_level() -> None:
    """alpha < 1/(n+1) means k > n: no finite threshold covers at 1 - alpha."""
    scores = np.array([0.2, 0.4, 0.6, 0.8, 0.9])  # n = 5, need alpha >= 1/6
    with pytest.raises(ValueError, match="unattainable"):
        replicable_conformal_threshold(scores, 0.10, 0.05, 0.0)
    tau = replicable_conformal_threshold(scores, 0.20, 0.05, 0.0)  # k = 5 <= n
    assert np.isfinite(tau)
    with pytest.raises(ValueError, match="unattainable"):
        expected_min_coverage(0.10, 5, 2)
    with pytest.raises(ValueError, match="unattainable"):
        agreement_experiment(n=5, alpha=0.10)


def test_fail_closed_empty_and_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        replicable_conformal_threshold(np.array([]), ALPHA, 0.1, 0.0)
    with pytest.raises(ValueError, match="finite"):
        replicable_conformal_threshold(np.array([np.nan, np.inf, -np.inf]), ALPHA, 0.1, 0.0)
    with pytest.raises(ValueError, match="finite"):
        pilot_density(np.array([]), ALPHA, 0.1)


def test_fail_closed_pilot_edges() -> None:
    rng = np.random.default_rng(2)
    good = rng.random(100)
    with pytest.raises(ValueError, match="even"):
        pilot_density(good[:99], ALPHA, 0.05)
    with pytest.raises(ValueError, match="window"):
        pilot_density(good, ALPHA, 0.0)
    with pytest.raises(ValueError, match="accuracy"):
        pilot_density(good, ALPHA, 0.05, accuracy=1.0)
    with pytest.raises(ValueError, match="accuracy"):
        pilot_density(good, ALPHA, 0.05, accuracy=-0.1)
    tiny = rng.random(4)  # m = 2 halves; alpha = 0.01 needs k_p = 3 > m
    with pytest.raises(ValueError, match="pilot half too small"):
        pilot_density(tiny, 0.01, 0.05)


def test_fail_closed_grid_width_and_protocol_params() -> None:
    with pytest.raises(ValueError, match="rho"):
        grid_width(ALPHA, 1_000, 0.0, 1.0)
    with pytest.raises(ValueError, match="rho"):
        grid_width(ALPHA, 1_000, 1.5, 1.0)
    with pytest.raises(ValueError, match="f_hat"):
        grid_width(ALPHA, 1_000, RHO, 0.0)
    with pytest.raises(ValueError, match="f_hat"):
        grid_width(ALPHA, 1_000, RHO, float("nan"))
    with pytest.raises(ValueError, match="kappa_hat"):
        grid_width(ALPHA, 1_000, RHO, 1.0, kappa_hat=0.5)
    with pytest.raises(ValueError, match="grid rule"):
        grid_width(ALPHA, 1_000, RHO, 1.0, rule="round_down")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="eps"):
        epsilon_protocol_grid_width(0.0, 1.0)
    with pytest.raises(ValueError, match="f_max"):
        epsilon_protocol_grid_width(0.05, -1.0)
    with pytest.raises(ValueError, match="rho"):
        protocol_sample_requirement(ALPHA, 0.05, 1.2)
    with pytest.raises(ValueError, match="kappa"):
        protocol_sample_requirement(ALPHA, 0.05, RHO, kappa=0.9)
    with pytest.raises(ValueError, match="delta"):
        protocol_sample_requirement(ALPHA, 0.05, RHO, delta=0.0)


def test_fail_closed_theorem3_and_4_inputs() -> None:
    with pytest.raises(ValueError, match="eps <= min"):
        sample_lower_bound(0.10, 0.03, RHO)  # 0.03 > min(0.1, 0.9)/4 = 0.025
    with pytest.raises(ValueError, match="delta"):
        sample_lower_bound(0.10, 0.02, RHO, delta=0.1)  # > 1/16
    with pytest.raises(ValueError, match="eps"):
        seedless_sample_requirement(0.0, 0.05)
    with pytest.raises(ValueError, match="delta"):
        seedless_sample_requirement(0.05, 1.0)
    with pytest.raises(ValueError, match="eps"):
        seedless_experiment(n=4_000, alpha=0.10, eps=0.10)  # eps must be < min(a, 1-a)
    with pytest.raises(ValueError, match="rho"):
        asymptotic_mismatch_limit(0.0)


def test_fail_closed_experiment_params() -> None:
    with pytest.raises(ValueError, match="n_pairs"):
        agreement_experiment(n=500, n_pairs=0)
    with pytest.raises(ValueError, match="pool_size"):
        agreement_experiment(n=500, n_pairs=1, pool_size=0)
    with pytest.raises(ValueError, match="n_trials"):
        validity_cost_experiment(n=500, n_trials=0)
    with pytest.raises(ValueError, match="at least two"):
        selective_recalibration_experiment(n=500, m=1)
    with pytest.raises(ValueError, match="m must be >= 1"):
        selective_recalibration_bound(0, RHO)
    with pytest.raises(ValueError, match="beta"):
        agreement_experiment(n=500, n_pairs=1, beta=-1.0)
    with pytest.raises(ValueError, match="offset"):
        validity_cost_experiment(n=500, n_trials=1, beta=0.1, offset=0.2)


# ---------------------------------------------------------------------------
# Closed-form theory (Theorems 2-4, Corollaries 1-4, Propositions 2-3)
# ---------------------------------------------------------------------------


def test_b_n_spread_formula_and_monotonicity() -> None:
    n = 20_000
    expected = np.sqrt(2.0 * 0.1 * 0.9 / n) + np.sqrt(2.0) / n
    assert b_n_spread(0.10, n) == pytest.approx(float(expected), rel=1e-12)
    assert b_n_spread(0.10, 4 * n) < b_n_spread(0.10, n)
    with pytest.raises(ValueError, match="n must be"):
        b_n_spread(0.10, 0)


def test_grid_width_rules_match_paper_formulas() -> None:
    alpha, n, rho, f_hat, kh = 0.10, 20_000, 0.10, 1.0, 1.5
    prop2 = grid_width(alpha, n, rho, f_hat, kh, rule="prop2")
    appx_g = grid_width(alpha, n, rho, f_hat, kh, rule="appendix_g")
    assert prop2 == pytest.approx(2.0 * kh * b_n_spread(alpha, n) / (f_hat * rho), rel=1e-12)
    assert appx_g == pytest.approx(
        kh * np.sqrt(2 * alpha * (1 - alpha) / n) / (f_hat * rho), rel=1e-12
    )
    assert prop2 == pytest.approx(0.09212132034355963, rel=1e-9)  # pinned
    assert appx_g < prop2  # leading-term rule is half as wide at equal kappa_hat


def test_disagreement_bound_theorem2_shape() -> None:
    alpha, n = 0.10, 20_000
    beta = grid_width(alpha, n, RHO, 1.0, 1.5)
    bound = 0.03333333333333334  # B_n / (f_min beta) at the plug-in width, kappa_hat=1.5
    got = disagreement_bound(alpha, n, beta, 1.0, 0.09)
    assert got == pytest.approx(bound, abs=1e-6)
    assert got <= RHO  # at the protocol width the bound beats the target
    # monotone in the grid width and clipped to a probability
    assert disagreement_bound(alpha, n, 2 * beta, 1.0, 0.09) < got
    assert disagreement_bound(alpha, n, 1e-9, 1.0, 0.09) == 1.0


def test_protocol_sample_requirement_remark2_anchor() -> None:
    """Remark 2: exact-Beta (Cor 1) analysis gives 7.2e5 at (0.1, 0.02, 0.1, 0.05), kappa=1."""
    req = protocol_sample_requirement(0.1, 0.02, 0.1, kappa=1.0, delta=0.05)
    assert req == 720_000
    # decreasing in eps and rho; the certified constants are conservative
    assert protocol_sample_requirement(0.1, 0.04, 0.1) < req
    assert protocol_sample_requirement(0.1, 0.02, 0.2) < req


def test_sample_lower_bound_theorem3_below_upper_requirement() -> None:
    lb = sample_lower_bound(0.1, 0.02, 0.1)  # ceil(9/4096 * 0.09 / 4e-6) at delta = 0
    assert lb == 50
    lb16 = sample_lower_bound(0.1, 0.02, 0.1, delta=1.0 / 16.0)
    assert lb16 < lb  # (1 - 8 delta)^2 shrinks the constant
    # no-threshold-method-pays-less: lower bound <= any valid upper requirement
    assert lb <= protocol_sample_requirement(0.1, 0.02, 0.1)


def test_replicability_size_floor_corollary4() -> None:
    alpha, n, rho = 0.10, 20_000, 0.10
    floor = replicability_size_floor(alpha, n, rho)
    assert floor == pytest.approx((3.0 / 70.0) * np.sqrt(alpha * (1 - alpha) / n) / rho, rel=1e-12)
    assert replicability_size_floor(alpha, 4 * n, rho) < floor  # decays as 1/sqrt(n)
    assert replicability_size_floor(alpha, n, rho / 2) > floor  # grows as 1/rho


def test_asymptotic_mismatch_limit_proposition3() -> None:
    lim = asymptotic_mismatch_limit(0.1)
    assert lim == pytest.approx(0.039894228040143274, rel=1e-9)  # pinned
    assert lim <= 2.0 * 0.1 / 5.0  # paper: <= rho sqrt(2/pi)/2 < 2 rho / 5
    assert asymptotic_mismatch_limit(0.5) > lim
    assert asymptotic_mismatch_limit(1.0) < 1.0


def test_selective_recalibration_bound_formula() -> None:
    assert selective_recalibration_bound(20, 0.01) == pytest.approx(0.19)  # paper F5 value
    assert selective_recalibration_bound(20, 0.1) == 1.0  # vacuous at rho=.1, M=20 (F5)
    assert selective_recalibration_bound(2, 0.05) == pytest.approx(0.05)
    assert selective_recalibration_bound(1, 0.5) == 0.0


def test_expected_min_coverage_m1_is_exact_beta_mean() -> None:
    alpha, n = 0.10, 20_000
    k = int(np.ceil((1 - alpha) * (n + 1)))
    mu_n = k / (n + 1)
    assert expected_min_coverage(alpha, n, 1) == pytest.approx(mu_n, abs=1e-7)


def test_expected_min_coverage_matches_beta_monte_carlo() -> None:
    """Eq. (19) by quadrature vs seeded Beta(k, n+1-k) min-of-M Monte Carlo."""
    alpha, n, m = 0.10, 999, 20
    k = int(np.ceil((1 - alpha) * (n + 1)))
    draws = beta_dist.rvs(k, n + 1 - k, size=(200_000, m), random_state=11)
    mc = float(draws.min(axis=1).mean())
    exact = expected_min_coverage(alpha, n, m)
    assert exact == pytest.approx(mc, abs=2e-4)
    assert exact < k / (n + 1)  # selection strictly hurts
    assert expected_min_coverage(alpha, n, 40) < exact  # decreasing in M


def test_expected_max_normal_known_values() -> None:
    assert expected_max_normal(1) == pytest.approx(0.0, abs=1e-9)
    assert expected_max_normal(2) == pytest.approx(1.0 / np.sqrt(np.pi), abs=1e-6)
    a20 = expected_max_normal(20)
    assert 1.8 < a20 < 1.95
    assert a20 > expected_max_normal(10)


def test_coverage_drop_approximation_eq20() -> None:
    alpha, n, m = 0.10, 20_000, 20
    approx = coverage_drop_approximation(alpha, n, m)
    exact = expected_min_coverage(alpha, n, m)
    assert approx == pytest.approx(exact, abs=1e-3)  # o(sigma_n) at fixed M
    k = int(np.ceil((1 - alpha) * (n + 1)))
    mu_n = k / (n + 1)
    sd_n = np.sqrt(mu_n * (1 - mu_n) / (n + 2))
    assert approx == pytest.approx(mu_n - expected_max_normal(m) * sd_n, rel=1e-9)


def test_conditional_coverage_band_feasibility() -> None:
    alpha, n, delta = 0.10, 20_000, 0.05
    band = conditional_coverage_band(alpha, n, delta, 0.01, 1.0, 1.0, 0.09)
    assert band.feasible
    assert band.lower <= 1.0 - alpha <= band.upper
    assert band.e_n == pytest.approx(np.sqrt(np.log(2 / delta) / (2 * n)) + 2 / n, rel=1e-12)
    # beta wider than the margin window: band infeasible (Theorem 2(ii) side condition)
    bad = conditional_coverage_band(alpha, n, delta, 0.12, 1.0, 1.0, 0.09)
    assert not bad.feasible
    with pytest.raises(ValueError, match="f_max"):
        conditional_coverage_band(alpha, n, delta, 0.01, 0.5, 1.0, 0.09)


def test_pilot_density_estimates_true_density() -> None:
    """Proposition 2 pilot on N(0, 1) scores: f_hat_L brackets the true f(q)."""
    rng = np.random.default_rng(5)
    pilot = rng.normal(size=4_000)
    f_true = float(norm.pdf(norm.ppf(1.0 - ALPHA)))  # 0.1755
    f_l = pilot_density(pilot, ALPHA, window=0.1, accuracy=0.25)
    assert 0.6 * f_true <= f_l <= 1.2 * f_true
    plain = pilot_density(pilot, ALPHA, window=0.1, accuracy=0.0)
    assert plain >= f_l  # the (1 + c) safeguard only lowers the estimate
    # PIT-uniform scores: density is exactly 1
    uni = pilot_density(rng.random(2_000), ALPHA, window=0.05, accuracy=0.0)
    assert 0.8 <= uni <= 1.2


# ---------------------------------------------------------------------------
# Agreement experiment (item 2): identity with empirical probability >= 1 - rho
# ---------------------------------------------------------------------------


def test_agreement_experiment_meets_rho_target() -> None:
    rep = agreement_experiment(n=20_000, alpha=ALPHA, rho=RHO, n_pairs=300, seed=23638)
    assert rep.identity_rate >= 1.0 - RHO  # the rho-replicability contract
    assert rep.mismatch_rate <= rep.disagreement_bound  # Theorem 2(i) respected
    assert rep.mismatch_rate <= RHO
    assert rep.set_identity_rate == rep.identity_rate  # identical tau <=> identical sets
    assert rep.pointwise_agreement >= rep.identity_rate
    assert 0.95 <= rep.identity_rate <= 1.0  # pinned: 0.9767, non-vacuous
    assert rep.asymptotic_mismatch == pytest.approx(0.02659615202676218, rel=1e-9)


def test_agreement_conditional_churn_is_one_cell_lemma1() -> None:
    rep = agreement_experiment(n=20_000, alpha=ALPHA, rho=RHO, n_pairs=300, seed=23638)
    one_cell = rep.beta * 50.0  # pool_size = 50 labels per context
    assert np.isfinite(rep.conditional_churn)
    assert 0.8 * one_cell <= rep.conditional_churn <= 1.2 * one_cell
    assert rep.mean_churn <= rep.conditional_churn


def test_agreement_kappa_hat_one_still_meets_target() -> None:
    rep = agreement_experiment(
        n=8_000, alpha=ALPHA, rho=RHO, n_pairs=150, seed=23642, kappa_hat=1.0
    )
    assert rep.identity_rate >= 1.0 - RHO
    assert rep.mismatch_rate <= rep.disagreement_bound
    assert rep.asymptotic_mismatch == pytest.approx(0.1 * np.sqrt(2 / np.pi) / 2, rel=1e-6)


def test_agreement_experiment_is_deterministic() -> None:
    def same(x: float, y: float) -> bool:
        return bool(np.isnan(x) and np.isnan(y)) or x == y

    a = agreement_experiment(n=4_000, alpha=ALPHA, rho=RHO, n_pairs=20, seed=77)
    b = agreement_experiment(n=4_000, alpha=ALPHA, rho=RHO, n_pairs=20, seed=77)
    for f in dataclasses.fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        assert same(float(va), float(vb)) if isinstance(va, float) else va == vb, f.name
    c = agreement_experiment(n=4_000, alpha=ALPHA, rho=RHO, n_pairs=20, seed=78)
    assert c.identity_rate >= 1.0 - RHO  # the contract holds for every seed
    assert c.beta == a.beta  # protocol constants are seed-independent


# ---------------------------------------------------------------------------
# Coverage validity and the price of replicability (items 3-4)
# ---------------------------------------------------------------------------


def test_coverage_survives_the_coarser_threshold() -> None:
    rep = validity_cost_experiment(n=20_000, alpha=ALPHA, rho=RHO, n_trials=200, seed=23639)
    assert rep.mean_coverage_recal >= 1.0 - ALPHA  # Theorem 2(ii): unconditional
    assert rep.mean_population_coverage_recal >= 1.0 - ALPHA
    assert rep.mean_coverage_standard >= 1.0 - ALPHA - 0.01
    assert rep.nested_dominance_rate == 1.0  # tau >= tau_tilde on every trial
    assert 0.93 <= rep.mean_coverage_recal <= 0.97  # pinned ~0.947


def test_coverage_inflation_tracks_half_cell_proposition3ii() -> None:
    rep = validity_cost_experiment(n=20_000, alpha=ALPHA, rho=RHO, n_trials=200, seed=23639)
    half_cell = rep.beta / 2.0  # E[F(tau)] - k/(n+1) ~ B_n / rho = beta/2 at kappa_hat=1.5,f=1
    assert 0.5 * half_cell <= rep.coverage_inflation <= 1.5 * half_cell
    assert rep.coverage_inflation <= rep.inflation_cap  # e_n(delta) + f_max beta
    assert rep.coverage_inflation >= rep.size_floor  # Corollary 4: everyone pays at least this


def test_size_cost_is_the_price_of_replicability() -> None:
    rep = validity_cost_experiment(n=20_000, alpha=ALPHA, rho=RHO, n_trials=200, seed=23639)
    assert rep.size_cost_ratio > 1.0  # replicability is never free (Corollary 4)
    assert rep.size_cost_ratio < 1.15  # pinned ~1.052, and it is not vacuous
    assert rep.mean_size_recal > rep.mean_size_standard
    assert rep.mean_size_standard == pytest.approx(50.0 * (1.0 - ALPHA), rel=0.02)


def test_eps_protocol_meets_certified_sample_requirement() -> None:
    """Corollary 1 configuration: beta = eps/(2 f_max), n exactly at the certificate."""
    beta_eps = epsilon_protocol_grid_width(0.06, 1.0)
    assert beta_eps == pytest.approx(0.03)
    rep = validity_cost_experiment(
        n=20_000,
        alpha=ALPHA,
        rho=0.20,
        n_trials=100,
        seed=23643,
        kappa_hat=1.0,
        beta=beta_eps,
        eps=0.06,
    )
    assert rep.sample_requirement == 20_000
    assert rep.requirement_met  # documented extra calibration-sample requirement satisfied
    assert rep.mean_coverage_recal >= 1.0 - ALPHA
    assert rep.mean_population_coverage_recal >= 1.0 - ALPHA
    assert rep.coverage_inflation <= 0.06 + 1e-9  # within one cell (eps protocol)


def test_certified_requirement_honestly_reported_when_unmet() -> None:
    rep = validity_cost_experiment(n=20_000, alpha=ALPHA, rho=RHO, n_trials=5, seed=101)
    assert not rep.requirement_met
    assert rep.sample_requirement > 20_000
    # marginal coverage is unconditional (Theorem 2(ii)) even when the certified
    # conditional band requirement is not met at this n
    assert rep.mean_coverage_recal >= 1.0 - ALPHA


# ---------------------------------------------------------------------------
# Anti-gaming demo (item 5): Corollary 3 selection attack
# ---------------------------------------------------------------------------


def test_gaming_standard_undercovers_while_recal_stays_valid() -> None:
    rep = selective_recalibration_experiment(
        n=20_000, alpha=ALPHA, rho=RHO, m=20, n_trials=100, seed=23640
    )
    nominal = 1.0 - ALPHA
    # standard split CP: min-of-M selection silently undercovers (paper F5)
    assert rep.standard_selected_coverage < nominal - 0.002
    assert rep.standard_undercover_gap > 0.002
    # exact Beta-law theory (eq. 19) matches the Monte Carlo
    assert rep.standard_selected_coverage == pytest.approx(rep.standard_expected_theory, abs=2e-3)
    assert rep.standard_expected_theory < rep.standard_honest_coverage
    # ReCal: every candidate is upward-rounded and marginally valid, so the
    # selected classifier stays at or above nominal
    assert rep.recal_selected_coverage >= nominal
    assert rep.recal_selected_coverage - rep.standard_selected_coverage >= 0.02
    assert rep.selection_bound_c_m == 1.0  # min(1, (M-1) rho) at rho=.1, M=20 (F5)
    assert rep.coverage_floor == pytest.approx(nominal - 1.0)


def test_gaming_recal_classifier_barely_moves() -> None:
    rep = selective_recalibration_experiment(
        n=20_000, alpha=ALPHA, rho=RHO, m=20, n_trials=100, seed=23640
    )
    assert rep.recal_stability >= 0.5  # pinned 0.87: all 20 draws give one classifier
    assert rep.recal_mean_distinct <= 1.5  # pinned ~1.13
    # selection can only shave rounding inflation off an honest ReCal draw
    assert rep.recal_selected_coverage <= rep.recal_honest_coverage + 1e-12
    assert rep.recal_selected_coverage >= rep.recal_honest_coverage - rep.beta - 1e-9


def test_gaming_tight_rho_gives_nonvacuous_bound() -> None:
    rep = selective_recalibration_experiment(
        n=8_000, alpha=ALPHA, rho=0.01, m=20, n_trials=50, seed=23644, kappa_hat=1.5
    )
    assert rep.selection_bound_c_m == pytest.approx(0.19)  # paper F5: (M-1) rho = .19
    assert rep.recal_stability >= 0.5
    assert rep.standard_undercover_gap > 0.002  # the attack still works on standard CP


# ---------------------------------------------------------------------------
# Seedless variant (item 6): Theorem 4 two-element adjacent list
# ---------------------------------------------------------------------------


def test_seedless_grid_confines_analysts_to_adjacent_classifiers() -> None:
    rep = seedless_experiment(n=40_000, alpha=ALPHA, eps=0.06, n_pairs=200, seed=23641)
    assert rep.adjacency_rate == 1.0  # paper F7: empirical mass 1.000
    assert rep.max_cell_gap <= 1  # two adjacent grid values at most
    assert rep.max_cell_gap == 1  # and adjacency is non-trivial: gaps do occur
    assert 0.2 <= rep.identity_rate <= 0.9  # pinned 0.455: identity is NOT guaranteed seedlessly
    assert rep.beta == pytest.approx(0.03)
    assert rep.sample_requirement == 32_791  # ceil(32 log(40) / 0.06^2)
    assert rep.requirement_met


def test_seedless_requirement_has_no_rho_factor() -> None:
    """Theorem 4: O(kappa^2 log(1/delta) / eps^2), optimally — no rho^-2 appears.

    Two answers cost neither a shared seed nor a 1/rho^2 sample factor: at a
    tight replicability target the single-answer (Corollary 1) certificate
    explodes while the seedless two-list certificate does not move.
    """
    req = seedless_sample_requirement(0.06, 0.05, kappa=1.0)
    assert req == 32_791
    assert req < protocol_sample_requirement(0.10, 0.06, 0.01, kappa=1.0, delta=0.05)
    # kappa enters quadratically (up to the ceiling), delta only logarithmically
    quad = seedless_sample_requirement(0.06, 0.05, kappa=2.0)
    assert 4 * req - 4 <= quad <= 4 * req
    assert seedless_sample_requirement(0.06, 0.005) < 2 * req


# ---------------------------------------------------------------------------
# Bench: honesty stamps, suggested keys, determinism
# ---------------------------------------------------------------------------


def test_bench_keys_values_and_honesty_stamps() -> None:
    row = bench_replicable_conformal(n=8_000, seed=23638)
    suggested = (
        "synthetic_repcon_agreement_rate",
        "synthetic_repcon_coverage",
        "synthetic_repcon_size_cost_ratio",
        "synthetic_repcon_gaming_undercover_standard",
        "synthetic_repcon_gaming_stability",
    )
    for key in suggested:
        assert key in row
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav", "return")
    for key in row:
        low = key.lower()
        assert all(tok not in low for tok in forbidden), key
    assert row["synthetic_dgp"] == "fixture"
    assert row["synthetic_claim"] == "research_metric_only"
    assert float(row["synthetic_repcon_agreement_rate"]) >= 0.9
    assert float(row["synthetic_repcon_coverage"]) >= 0.9
    assert float(row["synthetic_repcon_size_cost_ratio"]) > 1.0
    assert float(row["synthetic_repcon_gaming_undercover_standard"]) > 0.0
    assert float(row["synthetic_repcon_seedless_adjacency_rate"]) == 1.0


def test_bench_is_deterministic() -> None:
    a = bench_replicable_conformal(n=4_000, seed=5, n_pairs=30, n_trials=30, m=5)
    b = bench_replicable_conformal(n=4_000, seed=5, n_pairs=30, n_trials=30, m=5)
    assert a == b
