"""Tests for the sequential-LOB liquidity-tail-risk equilibrium module.

SYNTHETIC correctness tests only — the equilibrium here is the
Çetin-Lin-Livieri (2026, arXiv:2607.01198) marginal-cost fixed point; every
assertion is a numerical/consistency property, never market evidence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.microstructure.liquidity_tail_lob import (
    Belief,
    FixedPointError,
    NoiseSpec,
    SolverConfig,
    beta_belief,
    gaussian_noise,
    grid_belief,
    liquidity_tail_bench,
    simulate_learning,
    solve_marginal_cost,
    student_t_noise,
    theoretical_informed_exponent,
    theoretical_rho_sequence,
    truncnorm_belief,
    uniform_belief,
)

SIGMA = 2.0
CFG = SolverConfig(n_informed=2, x_max=10.0, n_x=81, z_max=30.0, n_z=241, tol=1e-8, max_iter=2000)


@pytest.fixture(scope="module")
def prior() -> Belief:
    return uniform_belief(0.0, 1.0, 201)


@pytest.fixture(scope="module")
def sol_t(prior: Belief):
    return solve_marginal_cost(prior, student_t_noise(3.0, SIGMA), CFG)


@pytest.fixture(scope="module")
def sol_g(prior: Belief):
    return solve_marginal_cost(prior, gaussian_noise(SIGMA), CFG)


# ---------------------------------------------------------------------------
# NoiseSpec validation + laws
# ---------------------------------------------------------------------------


def test_noise_spec_rejects_nonpositive_tail_index() -> None:
    with pytest.raises(ValueError, match="tail index"):
        student_t_noise(0.0, 1.0)
    with pytest.raises(ValueError, match="tail index"):
        student_t_noise(-2.5, 1.0)


def test_noise_spec_rejects_bad_sigma_and_kind() -> None:
    with pytest.raises(ValueError, match="sigma"):
        student_t_noise(3.0, 0.0)
    with pytest.raises(ValueError, match="kind"):
        NoiseSpec(kind="cauchy", nu=3.0, sigma=1.0)  # type: ignore[arg-type]


def test_noise_spec_gaussian_requires_inf_nu() -> None:
    with pytest.raises(ValueError, match="nu"):
        NoiseSpec(kind="gaussian", nu=3.0, sigma=1.0)
    g = gaussian_noise(1.0)
    assert g.nu == math.inf


def test_noise_pdf_normalizes_and_matches_logpdf() -> None:
    z = np.linspace(-60.0, 60.0, 8001)
    for nz in (student_t_noise(3.0, SIGMA), gaussian_noise(SIGMA)):
        mass = float(np.trapezoid(nz.pdf(z), z))
        assert mass == pytest.approx(1.0, abs=5e-4)
        assert np.allclose(np.log(nz.pdf(z[100])), nz.logpdf(z[100:101])[0])
        assert np.allclose(nz.sf(z), 1.0 - nz.cdf(z), atol=1e-8)


def test_noise_sample_seeded_deterministic() -> None:
    nz = student_t_noise(3.0, 1.5)
    a = nz.sample(np.random.default_rng(7), 20000)
    b = nz.sample(np.random.default_rng(7), 20000)
    np.testing.assert_array_equal(a, b)
    assert abs(float(np.mean(a))) < 0.1  # centered (se << 0.1 at this size)


# ---------------------------------------------------------------------------
# Belief validation + tail functionals
# ---------------------------------------------------------------------------


def test_belief_rejects_bad_grids() -> None:
    v = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError):
        Belief(v=v, w=np.ones(10))
    with pytest.raises(ValueError, match="non-negative"):
        Belief(v=v, w=-np.ones(11))
    with pytest.raises(ValueError, match="positive total"):
        Belief(v=v, w=np.zeros(11))
    with pytest.raises(ValueError, match="strictly increasing"):
        Belief(v=v[::-1], w=np.ones(11))


def test_uniform_belief_tail_functionals_closed_form() -> None:
    b = uniform_belief(0.0, 1.0, 401)
    y = np.array([0.2, 0.5, 0.8])
    phi, pi = b.tail_plus(y)
    # Pi+(y) ~ 1 - y, Psi+(y) = (1 + y) / 2 for U[0,1]
    np.testing.assert_allclose(pi, 1.0 - y, atol=2e-3)
    np.testing.assert_allclose(b.psi_plus(y), 0.5 * (1.0 + y), atol=2e-3)


def test_belief_endpoint_conventions() -> None:
    b = uniform_belief(0.0, 1.0, 101)
    # y > M: Pi+ = 0 and Psi+ convention = M ; y < m symmetric (eq. 3.6).
    # At y = M exactly the grid-endpoint mass is retained (discrete prior),
    # so the strict-interior check sits just above the boundary.
    _, pi_p = b.tail_plus(np.array([1.001, 1.5]))
    assert np.all(pi_p == 0.0)
    assert b.psi_plus(np.array([1.0, 1.5])) == pytest.approx(np.array([1.0, 1.0]))
    _, pi_m = b.tail_minus(np.array([-0.001, -0.5]))
    assert np.all(pi_m == 0.0)
    assert b.psi_minus(np.array([0.0, -0.5])) == pytest.approx(np.array([0.0, 0.0]))


def test_grid_belief_trapezoid_normalization() -> None:
    v = np.linspace(0.0, 1.0, 51)
    b = grid_belief(v, v * (1.0 - v) + 0.5)
    assert float(b.w.sum()) == pytest.approx(1.0)
    assert b.mean == pytest.approx(float(np.sum(b.v * b.w)))
    with pytest.raises(ValueError):
        grid_belief(v, -np.ones_like(v))


def test_beta_and_truncnorm_beliefs(prior: Belief) -> None:
    bb = beta_belief(2.0, 5.0, 0.0, 1.0, 101)
    assert 0.2 < bb.mean < 0.5  # Beta(2,5) mean = 2/7 ~ 0.286
    tb = truncnorm_belief(0.5, 0.2, 0.0, 1.0, 101)
    assert abs(tb.mean - 0.5) < 0.05
    assert tb.std < prior.std


def test_posterior_update_concentrates_and_preserves_support() -> None:
    b = uniform_belief(0.0, 1.0, 101)
    ll = -0.5 * ((b.v - 0.7) / 0.05) ** 2
    post = b.posterior(ll)
    assert post.mean == pytest.approx(0.7, abs=0.03)
    assert post.mass_within(0.7, 0.05) == pytest.approx(0.6827, abs=0.05)  # +/-1 sigma
    assert post.mass_within(0.7, 0.15) > 0.95
    # support preserved: prior zeros stay zero
    w = np.where(b.v <= 0.5, 1.0, 0.0)
    b2 = Belief(v=b.v, w=w)
    post2 = b2.posterior(np.zeros_like(b.v))
    assert np.all(post2.w[b2.v > 0.5] == 0.0)


def test_posterior_fails_closed_on_total_collapse() -> None:
    b = uniform_belief(0.0, 1.0, 51)
    with pytest.raises(ValueError, match="collapsed"):
        b.posterior(np.full(51, -np.inf))


# ---------------------------------------------------------------------------
# SolverConfig validation + solver fail-closed
# ---------------------------------------------------------------------------


def test_solver_config_validation() -> None:
    with pytest.raises(ValueError):
        SolverConfig(n_informed=1)
    with pytest.raises(ValueError):
        SolverConfig(n_x=100)  # must be odd
    with pytest.raises(ValueError):
        SolverConfig(z_max=5.0, x_max=10.0)
    with pytest.raises(ValueError):
        SolverConfig(damping=0.0)


def test_unconverged_solution_strict_accessor(prior: Belief) -> None:
    cfg = SolverConfig(n_informed=2, x_max=8.0, n_x=41, z_max=20.0, n_z=121, tol=1e-14, max_iter=2)
    sol = solve_marginal_cost(prior, student_t_noise(3.0, 1.0), cfg)
    assert not sol.converged
    with pytest.raises(FixedPointError):
        sol.spread()
    with pytest.raises(FixedPointError):
        sol.crossover_depth()
    with pytest.raises(FixedPointError):
        sol.informed_share(1.0)


def test_degenerate_prior_reduces_to_constant_F() -> None:
    # All mass on v = M: Psi+ = M everywhere -> h = M -> F = M (closed form).
    near_point = Belief(v=np.linspace(0.0, 1.0, 11), w=np.r_[np.full(10, 1e-9), 1.0])
    cfg = SolverConfig(n_informed=2, x_max=4.0, n_x=41, z_max=12.0, n_z=121, tol=1e-9, max_iter=200)
    sol = solve_marginal_cost(near_point, gaussian_noise(1.0), cfg)
    sol.require_solution()
    np.testing.assert_allclose(sol.F, np.full_like(sol.x, 1.0), atol=1e-4)


def test_bad_F0_rejected(prior: Belief) -> None:
    with pytest.raises(ValueError, match="one value per x-grid"):
        solve_marginal_cost(prior, gaussian_noise(1.0), CFG, F0=np.zeros(10))
    with pytest.raises(ValueError, match="finite"):
        solve_marginal_cost(prior, gaussian_noise(1.0), CFG, F0=np.full(CFG.n_x, np.nan))


# ---------------------------------------------------------------------------
# Equilibrium fixed point: shape, monotonicity, spread
# ---------------------------------------------------------------------------


def test_fixed_point_converges_monotone_bounded(sol_t) -> None:
    sol_t.require_solution()
    assert sol_t.max_resid < 1e-6
    assert np.all(np.diff(sol_t.F) > 0.0)  # strictly increasing (eq. 3.2)
    assert sol_t.F[0] > sol_t.belief.m and sol_t.F[-1] < sol_t.belief.M


def test_fixed_point_symmetry_center(prior: Belief) -> None:
    # Uniform prior symmetric about 0.5 -> F(0) ~ E[V] = 0.5.
    sol = solve_marginal_cost(
        prior,
        student_t_noise(3.0, SIGMA),
        SolverConfig(
            n_informed=2, x_max=10.0, n_x=81, z_max=30.0, n_z=241, tol=1e-8, max_iter=2000
        ),
    )
    i0 = CFG.n_x // 2
    assert sol.F[i0] == pytest.approx(0.5, abs=0.05)


def test_endogenous_bid_ask_spread(sol_t, sol_g) -> None:
    # Lemma 3.1(iii): phi+(0) > phi-(0), a strictly positive spread.
    assert sol_t.spread() > 0.0
    assert sol_g.spread() > 0.0
    assert sol_t.h_plus[CFG.n_x // 2] > sol_t.h_minus[CFG.n_x // 2]


def test_price_schedule_properties(sol_t) -> None:
    i0 = CFG.n_x // 2
    assert float(sol_t.price(0.0)) == pytest.approx(float(sol_t.h_plus[i0]), abs=1e-6)
    assert float(sol_t.price(-0.25)) == pytest.approx(float(sol_t.h_minus[i0 - 1]), abs=1e-3)
    assert sol_t.price(5.0) > sol_t.price(0.5)  # non-decreasing limit prices
    assert sol_t.price(-5.0) < sol_t.price(-0.5)
    assert float(sol_t.price(0.0)) - float(sol_t.price(-0.25)) == pytest.approx(
        sol_t.spread(), abs=0.01
    )


def test_informed_demand_endpoints_and_inverse(sol_t) -> None:
    x = sol_t.informed_demand(np.array([0.2, 0.5, 0.8]))
    assert np.all(np.diff(x) > 0.0)  # F^{-1} increasing
    mid = float(sol_t.informed_demand(np.array([0.5]))[0])
    assert mid == pytest.approx(0.0, abs=0.6)
    assert math.isinf(float(sol_t.informed_demand(np.array([1.0]))[0]))
    assert math.isinf(float(sol_t.informed_demand(np.array([0.0]))[0]))
    with pytest.raises(ValueError):
        sol_t.informed_demand(np.array([1.5]))


def test_far_tail_book_monotone(sol_t, sol_g) -> None:
    # Prop. 5.2 / Cor. 5.3: monotonicity is recovered in the tails.
    assert sol_t.far_tail_monotone(0.7)
    assert sol_g.far_tail_monotone(0.7)
    with pytest.raises(ValueError):
        sol_t.far_tail_monotone(1.5)


def test_empirical_tail_exponent_finite_and_negative(sol_t, sol_g) -> None:
    rho_t = sol_t.empirical_tail_exponent("ask")
    rho_g = sol_g.empirical_tail_exponent("bid")
    assert -2.0 < rho_t < 0.0
    assert -2.0 < rho_g < 0.0


def test_informed_exceedance_monotone_decay(sol_t) -> None:
    y = np.array([0.5, 1.0, 2.0, 4.0, 6.0])
    ex = sol_t.informed_exceedance(y)
    assert np.all(np.diff(ex) < 0.0)
    assert np.all((ex > 0.0) & (ex <= 1.0))


# ---------------------------------------------------------------------------
# Tail-risk comparative statics: the paper's headline results
# ---------------------------------------------------------------------------


def test_crossover_deeper_under_heavy_tails(sol_t, sol_g) -> None:
    # Paper's core comparative static: heavier uninformed tails keep large
    # trades plausibly uninformed out to deeper depth -> crossover is farther.
    c_t = sol_t.crossover_depth()
    c_g = sol_g.crossover_depth()
    assert math.isfinite(c_t) and math.isfinite(c_g)
    assert c_t > c_g


def test_informed_share_rises_in_far_tail(sol_t, sol_g) -> None:
    y = np.array([2.0, 4.0, 6.0, 8.0])
    share_t = sol_t.informed_share(y)
    share_g = sol_g.informed_share(y)
    assert np.all((share_t >= 0.0) & (share_t <= 1.0))
    # informed share of marginal flow eventually dominates (Prop. 5.1 spirit)
    assert share_t[-1] > share_t[0]
    assert share_g[-1] > share_g[0]
    # at matched depth, gaussian noise is implausible sooner
    assert share_g[-1] > share_t[-1]


def test_conditional_informed_share_far_tail(sol_t) -> None:
    # Prop. 5.1: posterior on {X* >= y} given flow >= y concentrates at 1.
    c_mid = sol_t.conditional_informed_share(1.0)
    c_far = sol_t.conditional_informed_share(8.0)
    assert 0.0 <= c_mid <= 1.0 and 0.0 <= c_far <= 1.0
    assert c_far >= c_mid - 1e-6
    with pytest.raises(ValueError):
        sol_t.conditional_informed_share(-1.0)


def test_heavy_tail_flattens_price_impact(sol_t, sol_g) -> None:
    # Deep asks cost less marginal price under t-noise (ambiguous large trades).
    assert sol_t.price(6.0) < sol_g.price(6.0)


def test_n_informed_shifts_fixed_point(prior: Belief) -> None:
    cfg3 = SolverConfig(
        n_informed=4, x_max=10.0, n_x=81, z_max=30.0, n_z=241, tol=1e-8, max_iter=2000
    )
    sol4 = solve_marginal_cost(prior, student_t_noise(3.0, SIGMA), cfg3)
    sol4.require_solution()
    assert np.all(np.diff(sol4.F) > 0.0)


def test_warm_start_not_slower(sol_t) -> None:
    sol2 = solve_marginal_cost(sol_t.belief, sol_t.noise, CFG, F0=sol_t.F)
    sol2.require_solution()
    assert sol2.n_iter <= sol_t.n_iter


# ---------------------------------------------------------------------------
# Theory exponents (Lemma 3.3 / Thm. 5.1 / Cor. 5.1)
# ---------------------------------------------------------------------------


def test_rho_sequence_period_one_closed_form() -> None:
    # Uniform prior endpoint (kappa=1), N=2: rho_1 = -1/(1 + (N-1)/N) = -2/3.
    rho = theoretical_rho_sequence(1.0, 2, 3.0, 1)
    assert rho[0] == pytest.approx(-2.0 / 3.0)


def test_rho_sequence_recursion_and_validation() -> None:
    rho = theoretical_rho_sequence(1.0, 2, 3.0, 5)
    assert rho.shape == (5,)
    assert np.all(rho < 0.0)
    with pytest.raises(ValueError):
        theoretical_rho_sequence(0.0, 2, 3.0, 3)
    with pytest.raises(ValueError):
        theoretical_rho_sequence(1.0, 1, 3.0, 3)
    with pytest.raises(ValueError):
        theoretical_rho_sequence(1.0, 2, 0.0, 3)


def test_informed_exponent_consistency() -> None:
    a1 = theoretical_informed_exponent(1.0, 2, 3.0, 1)
    rho1 = theoretical_rho_sequence(1.0, 2, 3.0, 1)[0]
    # kappa=1, alpha_1=1 -> psi'/(1-psi') = 1 -> exponent = rho_1 = -2/3.
    assert a1 == pytest.approx(rho1)
    with pytest.raises(ValueError):
        theoretical_informed_exponent(1.0, 2, 3.0, 0)


# ---------------------------------------------------------------------------
# Sequential learning: posterior consistency + spread persistence
# ---------------------------------------------------------------------------


def test_learning_run_posterior_consistency(prior: Belief) -> None:
    run = simulate_learning(prior, student_t_noise(3.0, SIGMA), CFG, v0=0.7, n_periods=5, rng=11)
    assert run.posterior_std[-1] < run.posterior_std[0]
    assert run.posterior_consistency_error() < 0.05
    assert run.mass_within_eps[-1] > run.mass_within_eps[0]


def test_learning_run_validation_fail_closed(prior: Belief) -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        simulate_learning(prior, student_t_noise(3.0, SIGMA), CFG, v0=1.0, n_periods=2, rng=0)
    with pytest.raises(ValueError):
        simulate_learning(prior, student_t_noise(3.0, SIGMA), CFG, v0=0.5, n_periods=0, rng=0)
    with pytest.raises(ValueError):
        simulate_learning(
            prior, student_t_noise(3.0, SIGMA), CFG, v0=0.5, n_periods=2, rng=0, eps=-1.0
        )


def test_learning_run_determinism(prior: Belief) -> None:
    cfg = SolverConfig(n_informed=2, x_max=8.0, n_x=41, z_max=20.0, n_z=121, tol=1e-7, max_iter=800)
    r1 = simulate_learning(prior, gaussian_noise(SIGMA), cfg, v0=0.6, n_periods=3, rng=5)
    r2 = simulate_learning(prior, gaussian_noise(SIGMA), cfg, v0=0.6, n_periods=3, rng=5)
    np.testing.assert_array_equal(r1.y, r2.y)
    np.testing.assert_array_equal(r1.posterior_mean, r2.posterior_mean)
    np.testing.assert_array_equal(r1.spread, r2.spread)


def test_spread_persistence_ratio_well_defined(prior: Belief) -> None:
    run = simulate_learning(prior, student_t_noise(3.0, SIGMA), CFG, v0=0.7, n_periods=4, rng=3)
    ratio = run.spread_persistence()
    assert math.isfinite(ratio) and ratio >= 0.0


def test_learning_run_propagates_unconverged(prior: Belief) -> None:
    cfg = SolverConfig(n_informed=2, x_max=8.0, n_x=41, z_max=20.0, n_z=121, tol=1e-15, max_iter=1)
    with pytest.raises(FixedPointError):
        simulate_learning(prior, student_t_noise(3.0, 1.0), cfg, v0=0.5, n_periods=1, rng=0)


# ---------------------------------------------------------------------------
# Bench: flat SYNTHETIC dict contract
# ---------------------------------------------------------------------------


def test_bench_keys_and_values() -> None:
    out = liquidity_tail_bench(seed=0, n_periods=4)
    expected = {
        "synthetic_fixedpoint_max_resid_t",
        "synthetic_fixedpoint_max_resid_gauss",
        "synthetic_crossover_depth_t",
        "synthetic_crossover_depth_gauss",
        "synthetic_informed_dominance_size",
        "synthetic_spread_persistence_ratio",
        "synthetic_posterior_consistency_error",
        "synthetic_crossover_ratio_t_over_gauss",
        "synthetic_tail_rho_theory_t1",
        "synthetic_informed_share_deep_t",
        "synthetic_informed_share_deep_gauss",
    }
    assert expected <= set(out)
    assert all(k.startswith("synthetic_") for k in out)
    assert out["synthetic_converged_t"] == 1.0
    assert out["synthetic_converged_gauss"] == 1.0
    assert out["synthetic_fixedpoint_max_resid_t"] < 1e-6
    # headline comparative statics (SYNTHETIC)
    assert out["synthetic_crossover_depth_t"] > out["synthetic_crossover_depth_gauss"]
    assert out["synthetic_crossover_ratio_t_over_gauss"] > 1.0
    assert out["synthetic_h_deep_t"] < out["synthetic_h_deep_gauss"]
    assert out["synthetic_posterior_consistency_error"] < 0.05
    assert out["synthetic_tail_rho_theory_t1"] == pytest.approx(-2.0 / 3.0)


def test_bench_deterministic() -> None:
    a = liquidity_tail_bench(seed=9, n_periods=3)
    b = liquidity_tail_bench(seed=9, n_periods=3)
    assert a == b
