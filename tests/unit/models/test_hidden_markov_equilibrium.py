"""Hidden-Markov equilibrium (arXiv:2609.21684) -- SYNTHETIC correctness tests.

Seeded simulations and solver diagnostics only: the Wonham two-state belief
filter (Prop. 2.1) calibrated on simulated dividend paths (innovation
variance and PIT-mean proper-score checks), the pricing BVP (4.3)-(4.4)
verified against the Prop. 4.11 affine closed form at theta = 1 and the
Delta_g = 0 constant-price reduction, the Thm. 4.2 uniform bounds, factor
flatness sup|phi zeta - 1|, vol-of-belief shape (unimodal, sigma_S = sigma
at p in {0,1}), the Sec. 5.5 numeric pins (c_min = 0.005128,
sigma_S max ~= 35.6% near p* ~= 0.486, r_f in [1.17%, 6.92%]), the Prop. 5.5
leading skewness sign-flip across p* and tau -> 0 convergence to MC, the
Crank-Nicolson (5.10) surface against a shared-innovation Q Monte-Carlo,
and fail-closed edges.  Correctness material, never market evidence.
No Sharpe.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.hidden_markov_equilibrium import (
    HiddenMarkovEconomy,
    belief_drift,
    belief_vol,
    bench_hidden_markov_equilibrium,
    calibrate_figure_economy,
    calibrate_option_economy,
    closed_form_phi_theta1,
    mc_log_return_skewness,
    mc_log_returns_q,
    mc_option_call,
    option_call_cn,
    pricing_diagnostics,
    sample_skewness,
    simulate_dividends,
    solve_price_dividend_ratio,
    wonham_filter,
)

SEED = 20260930


@pytest.fixture(scope="module")
def option_econ() -> HiddenMarkovEconomy:
    return calibrate_option_economy()


@pytest.fixture(scope="module")
def option_sol(option_econ: HiddenMarkovEconomy):
    return solve_price_dividend_ratio(option_econ, n_grid=801)


@pytest.fixture(scope="module")
def theta1_sol():
    econ = calibrate_figure_economy(theta=1.0)
    return econ, solve_price_dividend_ratio(econ, n_grid=401)


# ---------------------------------------------------------------------------
# Economy construction — fail-closed validation
# ---------------------------------------------------------------------------


def test_economy_rejects_nonpositive_intensities() -> None:
    base = dict(
        g_high=0.05,
        g_low=-0.05,
        sigma=0.04,
        lam_up=0.05,
        lam_down=0.05,
        gamma=0.8,
        psi=5.0,
        delta=0.06,
    )
    with pytest.raises(ValueError):
        HiddenMarkovEconomy(**{**base, "lam_up": 0.0})
    with pytest.raises(ValueError):
        HiddenMarkovEconomy(**{**base, "lam_down": -0.01})


def test_economy_rejects_bad_sigma_g_ordering_psi1() -> None:
    base = dict(
        g_high=0.05,
        g_low=-0.05,
        sigma=0.04,
        lam_up=0.05,
        lam_down=0.05,
        gamma=0.8,
        psi=5.0,
        delta=0.06,
    )
    with pytest.raises(ValueError):
        HiddenMarkovEconomy(**{**base, "sigma": 0.0})
    with pytest.raises(ValueError):
        HiddenMarkovEconomy(**{**base, "g_high": -0.1, "g_low": 0.0})
    with pytest.raises(ValueError):
        HiddenMarkovEconomy(**{**base, "psi": 1.0})


def test_economy_rejects_wrong_side_gamma_psi() -> None:
    # theta = (1-gamma)/(1-1/psi) < 0 when gamma < 1 < psi sides disagree.
    with pytest.raises(ValueError):
        HiddenMarkovEconomy(
            g_high=0.05,
            g_low=-0.05,
            sigma=0.04,
            lam_up=0.05,
            lam_down=0.05,
            gamma=1.5,
            psi=2.0,
            delta=0.06,
        )


def test_economy_derived_quantities() -> None:
    e = calibrate_option_economy()
    assert e.theta == pytest.approx(0.25)
    assert e.eta == pytest.approx(0.02)
    assert e.q == pytest.approx(-3.0)
    # Paper Sec. 5.5 reports min c = 0.005128.
    assert e.c_min == pytest.approx(0.005128, abs=1e-6)
    assert e.positivity_holds
    assert e.stationary_high == pytest.approx(0.5)


def test_transition_matrix_exact_ctmc() -> None:
    e = calibrate_option_economy()
    t = e.transition_matrix(0.5)
    assert np.allclose(t.sum(axis=1), 1.0)
    assert np.all(t > 0.0)
    # Rows symmetric (lam_up == lam_down): P(stay) = (1 + rho)/2.
    rho = math.exp(-0.1 * 0.5)
    assert t[0, 0] == pytest.approx(0.5 * (1.0 + rho))
    assert t[0, 1] == pytest.approx(0.5 * (1.0 - rho))
    # dt -> 0 limit is the identity.
    t0 = e.transition_matrix(1e-9)
    assert np.allclose(t0, np.eye(2), atol=1e-6)


def test_belief_coefficient_shapes(option_econ: HiddenMarkovEconomy) -> None:
    p = np.linspace(0.0, 1.0, 11)
    b = belief_drift(p, option_econ)
    chi = belief_vol(p, option_econ)
    assert b[0] == pytest.approx(option_econ.lam_up)
    assert b[-1] == pytest.approx(-option_econ.lam_down)
    assert belief_drift(option_econ.stationary_high, option_econ) == pytest.approx(0.0)
    # chi vanishes at the endpoints and is unimodal with max at p = 1/2.
    assert chi[0] == 0.0 and chi[-1] == 0.0
    assert np.argmax(chi) == 5
    chi_fine = belief_vol(np.linspace(0.0, 1.0, 401), option_econ)
    assert float(np.max(chi_fine)) == pytest.approx(
        option_econ.delta_g / (4.0 * option_econ.sigma), abs=1e-6
    )


# ---------------------------------------------------------------------------
# Wonham filter on a simulated dividend stream
# ---------------------------------------------------------------------------


def test_simulate_dividends_deterministic_and_valid(
    option_econ: HiddenMarkovEconomy,
) -> None:
    s1 = simulate_dividends(option_econ, 500, 0.02, SEED)
    s2 = simulate_dividends(option_econ, 500, 0.02, SEED)
    assert np.array_equal(s1.levels, s2.levels)
    assert np.array_equal(s1.states, s2.states)
    assert set(np.unique(s1.states)) <= {0, 1}
    assert np.all(s1.levels > 0.0)
    assert np.allclose(np.diff(np.log(s1.levels)), s1.log_increments)


def test_simulate_dividends_fail_closed(option_econ: HiddenMarkovEconomy) -> None:
    with pytest.raises(ValueError):
        simulate_dividends(option_econ, 1, 0.02, SEED)
    with pytest.raises(ValueError):
        simulate_dividends(option_econ, 10, 0.0, SEED)
    with pytest.raises(ValueError):
        simulate_dividends(option_econ, 10, 0.02, SEED, x0=1.5)


def test_wonham_filter_tracks_state(option_econ: HiddenMarkovEconomy) -> None:
    sim = simulate_dividends(option_econ, 4_000, 0.02, SEED, x0=1)
    path = wonham_filter(sim.levels, sim.dt, option_econ)
    assert np.all((path.filtered > 0.0) & (path.filtered < 1.0))
    assert np.all((path.predicted >= 0.0) & (path.predicted <= 1.0))
    corr = float(np.corrcoef(path.filtered, sim.states[1:])[0, 1])
    assert corr > 0.7
    assert math.isfinite(path.log_likelihood)


def test_wonham_filter_innovations_calibrated(
    option_econ: HiddenMarkovEconomy,
) -> None:
    """Predictive-standardized innovations: unit variance, PIT mean ~ 0.5."""
    sim = simulate_dividends(option_econ, 8_000, 0.02, SEED + 1)
    path = wonham_filter(sim.levels, sim.dt, option_econ)
    # var estimator se ~ sqrt(2/(n-1)) * var ~ 0.016; allow 6 se.
    assert np.var(path.innovations) == pytest.approx(1.0, abs=0.1)
    # Uniform-PIT mean se = 1/sqrt(12 n); allow 6 se ~ 0.0043 plus slack.
    assert np.mean(path.pit) == pytest.approx(0.5, abs=0.02)
    assert np.all((path.pit > 0.0) & (path.pit < 1.0))


def test_wonham_filter_fail_closed(option_econ: HiddenMarkovEconomy) -> None:
    good = np.exp(np.cumsum(0.001 * np.ones(50)))
    with pytest.raises(ValueError):
        wonham_filter([1.0], 0.02, option_econ)  # too short
    with pytest.raises(ValueError):
        wonham_filter(np.append(good, -1.0), 0.02, option_econ)  # nonpositive
    with pytest.raises(ValueError):
        wonham_filter(np.append(good, np.nan), 0.02, option_econ)
    with pytest.raises(ValueError):
        wonham_filter(good, 0.0, option_econ)  # dt <= 0
    with pytest.raises(ValueError):
        wonham_filter(good, 0.02, option_econ, p0=0.0)
    with pytest.raises(ValueError):
        wonham_filter(good, 0.02, option_econ, p0=1.0)


def test_wonham_filter_deterministic(option_econ: HiddenMarkovEconomy) -> None:
    sim = simulate_dividends(option_econ, 300, 0.02, SEED)
    a = wonham_filter(sim.levels, sim.dt, option_econ)
    b = wonham_filter(sim.levels, sim.dt, option_econ)
    assert np.array_equal(a.filtered, b.filtered)
    assert a.log_likelihood == b.log_likelihood


# ---------------------------------------------------------------------------
# Pricing BVP: exact reductions, bounds, monotonicity, residuals
# ---------------------------------------------------------------------------


def test_theta1_closed_form_matches_newton(theta1_sol) -> None:
    econ, sol = theta1_sol
    cf = closed_form_phi_theta1(econ, sol.p)
    assert sol.n_iter <= 2  # linear problem: Newton needs one step
    assert float(np.max(np.abs(sol.phi - cf))) < 1e-9


def test_theta1_closed_form_formula() -> None:
    econ = calibrate_figure_economy(theta=1.0)
    c0, c1 = float(econ.costate(0.0)), float(econ.costate(1.0))
    b = (c0 - c1) / (c0 * c1 + c0 * econ.lam_down + c1 * econ.lam_up)
    a = (1.0 + econ.lam_up * b) / c0
    p = np.linspace(0.0, 1.0, 7)
    assert np.allclose(closed_form_phi_theta1(econ, p), a + b * p)
    assert closed_form_phi_theta1(econ, 0.0) == pytest.approx(a)


def test_closed_form_theta1_fail_closed(option_econ: HiddenMarkovEconomy) -> None:
    with pytest.raises(ValueError):
        closed_form_phi_theta1(option_econ, 0.5)  # theta != 1


def test_zero_gap_reduces_to_constant_price() -> None:
    """Delta_g = 0: no learning, chi == 0 and phi = theta/c constant."""
    econ = HiddenMarkovEconomy(
        g_high=0.03,
        g_low=0.03,
        sigma=0.05,
        lam_up=0.05,
        lam_down=0.05,
        gamma=0.8,
        psi=5.0,
        delta=0.06,
    )
    sol = solve_price_dividend_ratio(econ, n_grid=201)
    c = econ.costate(0.5)
    expected = econ.theta / c
    assert np.allclose(sol.phi, expected, rtol=1e-6)
    assert np.allclose(sol.stock_vol(np.linspace(0, 1, 9)), econ.sigma)


def test_option_economy_solve_bounds_and_monotonicity(option_sol) -> None:
    sol = option_sol
    econ = sol.econ
    assert np.all(sol.phi > 0.0)
    # Thm. 4.2 uniform bounds theta/c_max <= phi <= theta/c_min.
    assert float(np.min(sol.phi)) >= econ.theta / econ.c_max - 1e-9
    assert float(np.max(sol.phi)) <= econ.theta / econ.c_min + 1e-9
    # eta > 0 -> phi strictly increasing (Thm. 4.10).
    assert np.all(np.diff(sol.phi) > 0.0)
    assert sol.residual_inf < 1e-9


def test_pricing_diagnostics_flat_factor(option_sol) -> None:
    d = pricing_diagnostics(option_sol)
    assert d["bc_residual_max"] < 1e-9
    # Factor flatness sup|phi zeta - 1| == ODE residual on the check grid.
    assert d["factor_flatness"] < 1e-5
    assert d["ode_residual_max"] < 1e-5
    assert d["phi_lower_bound"] <= d["phi_min"]
    assert d["phi_upper_bound"] >= d["phi_max"]


def test_vol_of_belief_shape(option_sol) -> None:
    """sigma_S unimodal interior max, sigma_S(0)=sigma_S(1)=sigma, >= sigma."""
    sol = option_sol
    d = pricing_diagnostics(sol, n_check=401)
    assert d["sigmaS_endpoint0"] == pytest.approx(sol.econ.sigma)
    assert d["sigmaS_endpoint1"] == pytest.approx(sol.econ.sigma)
    assert d["sigmaS_min"] >= sol.econ.sigma - 1e-9
    assert 0.0 < d["p_star"] < 1.0
    # Paper Sec. 5.5: sigma_S max ~= 35.6% attained near p* ~= 0.486.
    assert d["sigmaS_max"] == pytest.approx(0.356, abs=0.01)
    assert d["p_star"] == pytest.approx(0.486, abs=0.02)
    # Unimodal: increasing before p*, decreasing after.
    pg = np.linspace(0.0, 1.0, 801)
    sig = sol.stock_vol(pg)
    i_star = int(np.argmax(sig))
    assert np.all(np.diff(sig[: i_star + 1]) >= -1e-12)
    assert np.all(np.diff(sig[i_star:]) <= 1e-12)


def test_short_rate_range_matches_paper(option_sol) -> None:
    """Sec. 5.5 reports r_f spanning 1.17%-6.92% over the belief simplex."""
    rf = option_sol.marginal_short_rate(np.linspace(0.0, 1.0, 401))
    assert float(rf.min()) == pytest.approx(0.0117, abs=0.001)
    assert float(rf.max()) == pytest.approx(0.0692, abs=0.002)
    assert np.all(np.isfinite(rf))


def test_solve_fail_closed() -> None:
    # Assumption 4.1 violated: gamma > 1 with psi < 1 keeps theta > 0 but
    # (1-gamma) < 0 sends c(0) = kappa + (1-gamma)|g2| below zero.
    bad = HiddenMarkovEconomy(
        g_high=0.05,
        g_low=-0.5,
        sigma=0.04,
        lam_up=0.05,
        lam_down=0.05,
        gamma=1.5,
        psi=0.5,
        delta=0.01,
    )
    assert bad.theta > 0.0
    assert not bad.positivity_holds
    with pytest.raises(ValueError):
        solve_price_dividend_ratio(bad)
    e = calibrate_option_economy()
    with pytest.raises(ValueError):
        solve_price_dividend_ratio(e, n_grid=8)
    with pytest.raises(ValueError):
        solve_price_dividend_ratio(e, tol=0.0)
    with pytest.raises(ValueError):
        solve_price_dividend_ratio(e, max_iter=0)


def test_eta_nonpositive_gates_risk_neutral_objects() -> None:
    """gamma > 1, psi < 1 subregion: eta < 0 -> Sec. 5 objects refuse."""
    econ = HiddenMarkovEconomy(
        g_high=0.05,
        g_low=-0.05,
        sigma=0.04,
        lam_up=0.05,
        lam_down=0.05,
        gamma=1.5,
        psi=0.5,
        delta=0.06,
    )
    assert econ.eta < 0.0
    sol = solve_price_dividend_ratio(econ, n_grid=201)
    with pytest.raises(ValueError):
        sol.marginal_short_rate(0.5)
    with pytest.raises(ValueError):
        sol.rn_belief_drift(0.5)
    with pytest.raises(ValueError):
        mc_log_returns_q(sol, 0.5, 0.05, 8, 16, SEED)
    with pytest.raises(ValueError):
        option_call_cn(sol, strike=1.0, tau=0.1)


# ---------------------------------------------------------------------------
# Prop. 5.5 leading skewness expansion
# ---------------------------------------------------------------------------


def test_skewness_coefficient_sign_flip(option_sol) -> None:
    """sigma_S peaks near p* => skewness coefficient + below p*, - above."""
    sol = option_sol
    d = pricing_diagnostics(sol)
    assert sol.skewness_coefficient(0.3) > 0.0
    assert sol.skewness_coefficient(0.7) < 0.0
    assert abs(float(sol.skewness_coefficient(d["p_star"]))) < 0.5


def test_leading_skewness_sqrt_scaling(option_sol) -> None:
    sol = option_sol
    s1 = float(sol.leading_skewness(0.3, 0.01))
    s4 = float(sol.leading_skewness(0.3, 0.04))
    assert s4 == pytest.approx(2.0 * s1)


def test_skewness_expansion_vs_mc(option_sol) -> None:
    """MC skewness approaches the sqrt(tau) expansion as tau -> 0."""
    sol = option_sol
    lead = float(sol.leading_skewness(0.3, 0.01))
    mc, se = mc_log_return_skewness(sol, 0.3, tau=0.01, n_steps=256, n_paths=80_000, seed=SEED)
    assert abs(lead - mc) < max(0.25 * abs(lead), 4.0 * se)
    # Rel error shrinks with tau (o(sqrt(tau)) convergence).
    e_big, _ = mc_log_return_skewness(sol, 0.3, 0.04, 128, 60_000, SEED)
    e_sml, _ = mc_log_return_skewness(sol, 0.3, 0.005, 128, 60_000, SEED)
    r_big = abs(float(sol.leading_skewness(0.3, 0.04)) - e_big)
    r_sml = abs(float(sol.leading_skewness(0.3, 0.005)) - e_sml)
    assert r_sml < r_big


def test_sample_skewness_normal_and_fail_closed() -> None:
    rng = np.random.default_rng(SEED)
    sk, se = sample_skewness(rng.standard_normal(40_000))
    assert abs(sk) < 5.0 * se
    with pytest.raises(ValueError):
        sample_skewness(np.zeros(100))  # zero variance
    with pytest.raises(ValueError):
        sample_skewness([1.0, 2.0])


# ---------------------------------------------------------------------------
# European option PDE (5.10): Crank-Nicolson vs Monte-Carlo
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def call_surface(option_sol):
    return option_call_cn(option_sol, strike=1.0, tau=0.10, s0=1.0, n_x=121, n_p=61, n_t=50)


def test_option_cn_matches_mc(call_surface, option_sol) -> None:
    for p0 in (0.3, 0.5):
        cn = call_surface.price(1.0, p0)
        mc, se = mc_option_call(
            option_sol,
            p0,
            1.0,
            0.10,
            s0=1.0,
            n_steps=64,
            n_paths=60_000,
            seed=SEED + 7,
        )
        assert abs(cn - mc) < max(0.03 * mc, 5.0 * se)


def test_option_surface_shape_and_bounds(call_surface) -> None:
    v = call_surface.values
    assert v.shape == (121, 61)
    # CN + 7-point cross stencil is not exactly positivity-preserving: the
    # residual deep-OTM oscillation is ~3e-5 on this grid (module docstring
    # documents ~1e-4); allow 6x that measurement for solver variation.
    assert np.all(v >= -2e-4)
    # Call value nondecreasing in spot at fixed belief (tolerance covers
    # the ~1e-25 dust in the flat zero region and the documented ~1e-4
    # kink oscillation).
    for j in (10, 30, 50):
        col = v[:, j]
        assert np.all(np.diff(col) >= -2e-4)
    # Inside [0, S] bounds (rates can be small but nonnegative here).
    assert float(v.max()) <= math.exp(float(call_surface.x_grid[-1]))
    # Intrinsic lower bound near the right edge of the X grid.
    i = np.searchsorted(call_surface.x_grid, math.log(1.3))
    assert call_surface.price(1.3, 0.5) >= (1.3 - 1.0) - 0.02
    assert i > 0


def test_option_price_interp_and_fail_closed(call_surface, option_sol) -> None:
    with pytest.raises(ValueError):
        call_surface.price(1.0, 0.0)
    with pytest.raises(ValueError):
        call_surface.price(1.0, 1.5)
    with pytest.raises(ValueError):
        call_surface.price(1e6, 0.5)  # outside X grid
    with pytest.raises(ValueError):
        option_call_cn(option_sol, strike=0.0, tau=0.1)
    with pytest.raises(ValueError):
        option_call_cn(option_sol, strike=1.0, tau=0.0)
    with pytest.raises(ValueError):
        option_call_cn(option_sol, strike=1.0, tau=0.1, n_p=4)


def test_mc_option_deterministic(option_sol) -> None:
    a, _ = mc_option_call(option_sol, 0.5, 1.0, 0.1, n_steps=32, n_paths=8_000, seed=5)
    b, _ = mc_option_call(option_sol, 0.5, 1.0, 0.1, n_steps=32, n_paths=8_000, seed=5)
    assert a == b


# ---------------------------------------------------------------------------
# Bench helper
# ---------------------------------------------------------------------------


def test_bench_flat_float_dict_deterministic() -> None:
    b1 = bench_hidden_markov_equilibrium(seed=SEED)
    b2 = bench_hidden_markov_equilibrium(seed=SEED)
    assert b1 == b2  # bit-identical repeat
    assert all(isinstance(k, str) and isinstance(v, float) for k, v in b1.items())
    assert b1["synthetic_ode_residual_max"] < 1e-5
    assert b1["synthetic_sigmaS_max"] == pytest.approx(0.356, abs=0.01)
    assert b1["synthetic_filter_state_corr"] > 0.7
