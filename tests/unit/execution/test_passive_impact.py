"""Tests for passive-impact optimal execution (Barzykin et al. 2026).

All Monte-Carlo assertions are seeded SYNTHETIC correctness checks — never
market evidence.  No Sharpe-family metrics appear anywhere here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from quant_fund.execution.almgren_chriss import almgren_chriss_trajectory
from quant_fund.execution.passive_impact import (
    PassiveImpactSpec,
    aggressive_fill_price,
    aggressive_schedule,
    compare_vs_aggressive_baseline,
    expected_fill_time,
    expected_inventory_path,
    fill_intensity,
    fill_probability,
    fill_probability_curve,
    ofi_price_response,
    optimal_execution_plan,
    order_flow_imbalance,
    passive_impact_rate,
    repost_attempts,
    simulate_passive_execution,
    urgency_kappa,
    value_function_quote,
)

# Paper Table 1 (Barzykin et al. 2026) with delta measured in $0.01 ticks:
# k = 48 $^-1 -> 0.48 tick^-1; eta = 0.0028 $/s at delta = 0.
PAPER = dict(lam=1.4, k=0.48, eta=0.0028 / 0.01, sigma=0.003, phi=1e-5, alpha=1e-5)


def _spec(**kw: float) -> PassiveImpactSpec:
    return PassiveImpactSpec(**{**PAPER, **kw})


def _plan(spec: PassiveImpactSpec | None = None, q0: int = 20, T: float = 300.0, n: int = 120):
    return optimal_execution_plan(spec or _spec(), q0, T, n_grid=n)


# ---------------------------------------------------------------------------
# Fill model: closed forms.
# ---------------------------------------------------------------------------


def test_fill_intensity_closed_form() -> None:
    spec = _spec()
    d = np.array([0.0, 1.0, 5.0])
    got = fill_intensity(d, spec.lam, spec.k)
    np.testing.assert_allclose(got, spec.lam * np.exp(-spec.k * d), rtol=1e-14)
    assert fill_intensity(0.0, spec.lam, spec.k) == pytest.approx(spec.lam)


def test_fill_probability_closed_form() -> None:
    spec = _spec()
    d, dt = 3.0, 2.0
    lam_eff = spec.lam * math.exp(-spec.k * d)
    assert fill_probability(d, dt, spec.lam, spec.k) == pytest.approx(
        1.0 - math.exp(-lam_eff * dt), rel=1e-14
    )


def test_fill_probability_consistent_with_intensity() -> None:
    # p -> Lambda * dt as dt -> 0 (paper eq. 2.9 little-o limit).
    spec = _spec()
    d = 4.0
    lam_eff = float(fill_intensity(d, spec.lam, spec.k))
    p_small = float(fill_probability(d, 1e-4, spec.lam, spec.k))
    assert p_small == pytest.approx(lam_eff * 1e-4, rel=1e-4)


def test_expected_fill_time_closed_form() -> None:
    spec = _spec()
    d = np.array([0.0, 2.0, -1.0])
    np.testing.assert_allclose(
        expected_fill_time(d, spec.lam, spec.k), np.exp(spec.k * d) / spec.lam, rtol=1e-14
    )


def test_repost_attempts_inverse_of_fill_prob() -> None:
    spec = _spec()
    d, tau = 2.5, 0.5
    p = fill_probability(d, tau, spec.lam, spec.k)
    np.testing.assert_allclose(
        repost_attempts(d, tau, spec.lam, spec.k), 1.0 / np.asarray(p), rtol=1e-14
    )


def test_passive_impact_rate_closed_form() -> None:
    spec = _spec()
    d = np.array([0.0, 3.0])
    np.testing.assert_allclose(
        passive_impact_rate(d, spec.eta, spec.k), spec.eta * np.exp(-spec.k * d), rtol=1e-14
    )


def test_monotone_depth_fill_tradeoff() -> None:
    spec = _spec()
    depths = np.linspace(-1.0, 12.0, 50)
    probs = np.asarray(fill_probability(depths, 1.0, spec.lam, spec.k))
    impact = np.asarray(passive_impact_rate(depths, spec.eta, spec.k))
    assert np.all(np.diff(probs) < 0.0)
    assert np.all(np.diff(impact) < 0.0)


# ---------------------------------------------------------------------------
# OFI: Cont et al. linear response (paper eqs. 2.1-2.2).
# ---------------------------------------------------------------------------


def test_ofi_sign_convention() -> None:
    # A bid-side placement adds +1; an ask-side market order adds -1.
    ofi = order_flow_imbalance(
        limit_bid=3.0,
        cancel_bid=1.0,
        market_ask=2.0,
        limit_ask=4.0,
        cancel_ask=0.5,
        market_bid=1.5,
    )
    assert float(ofi) == pytest.approx(3.0 - 1.0 - 2.0 - 4.0 + 0.5 + 1.5)


def test_ofi_price_response_linear() -> None:
    spec = _spec(beta_ofi=0.02)
    ofi = np.array([-100.0, 0.0, 250.0])
    np.testing.assert_allclose(
        ofi_price_response(ofi, spec.beta_ofi), spec.beta_ofi * ofi, rtol=1e-14
    )
    np.testing.assert_allclose(ofi_price_response(ofi, 0.0), np.zeros(3))


# ---------------------------------------------------------------------------
# Optimal schedule, m == k closed form (paper Thm. 3.1).
# ---------------------------------------------------------------------------


def test_omega_matches_direct_matrix_exponential() -> None:
    # Pin the internal omega evaluation against a literal expm build.
    spec = _spec()
    q0, T = 8, 60.0
    q_desc = np.arange(q0, -1, -1, dtype=float)
    a = np.zeros((q0 + 1, q0 + 1))
    for i, q in enumerate(q_desc):
        a[i, i] = spec.k * spec.phi * q * q
        if i < q0:
            a[i, i + 1] = -(1.0 / math.e) * spec.lam * math.exp(-spec.k * (spec.eta / spec.lam) * q)
    b = np.exp(-spec.k * spec.alpha * q_desc**2)
    plan = _plan(spec, q0, T, n=20)
    for i in (0, 10, 20):
        tau = T - plan.times[i]
        w_ref = expm(-tau * a) @ b  # ordered q0..0
        theta_ref = np.log(w_ref[::-1]) / spec.k
        np.testing.assert_allclose(plan.theta_log_weights[i], theta_ref, rtol=1e-9, atol=1e-9)


def test_quote_equals_closed_form_components() -> None:
    spec = _spec()
    q0, T = 6, 50.0
    plan = _plan(spec, q0, T, n=10)
    i = 4
    theta = plan.theta_log_weights
    for q in (1, 3, 6):
        expected = 1.0 / spec.k + (theta[i, q] - theta[i, q - 1]) + (spec.eta / spec.lam) * q
        assert plan.quotes[i, q] == pytest.approx(expected, rel=1e-12)


def test_urgency_ordering_larger_inventory_quotes_tighter() -> None:
    # Paper Fig. 1: delta*(t, q) decreases in q — more remaining inventory
    # means more urgency, i.e. a quote closer to (or through) the midprice.
    plan = _plan()
    for i in (0, 30, 60):
        assert plan.quotes[i, 20] < plan.quotes[i, 5] < plan.quotes[i, 1]


def test_eta_shifts_quotes_deeper() -> None:
    # Paper Fig. 2: larger passive impact eta pushes the quote deeper.
    plan_lo = _plan(_spec(eta=0.0))
    plan_hi = _plan(_spec(eta=2 * PAPER["eta"]))
    i, q = 10, 20
    assert plan_hi.quotes[i, q] > plan_lo.quotes[i, q]


def test_terminal_quotes_terminal_condition() -> None:
    # At t = T, theta(q) = -alpha q^2 so delta*(T, q) = 1/k - alpha
    # (2q-1) + (eta/lam) q exactly.
    spec = _spec()
    plan = _plan(spec, 8, 60.0, n=20)
    q = 5
    expected = 1.0 / spec.k - spec.alpha * (2 * q - 1) + (spec.eta / spec.lam) * q
    assert plan.quotes[-1, q] == pytest.approx(expected, rel=1e-9)


def test_value_function_quote_matches_plan() -> None:
    spec = _spec()
    plan = _plan(spec, 10, 100.0, n=50)
    t, q = float(plan.times[10]), 4
    assert value_function_quote(t, q, spec, 10, 100.0) == pytest.approx(
        plan.quote_at(t, q), rel=1e-12
    )


def test_eta_zero_recovers_fill_only_quote() -> None:
    # Remark 3.3 / Guéant reduction: eta = 0 removes the impact term.
    spec0 = _spec(eta=0.0)
    plan = _plan(spec0, 6, 60.0, n=20)
    i, q = 5, 4
    theta = plan.theta_log_weights
    assert plan.quotes[i, q] == pytest.approx(
        1.0 / spec0.k + (theta[i, q] - theta[i, q - 1]), rel=1e-12
    )


def test_eta_zero_quote_independent_of_m() -> None:
    # With no impact, the m != k Lambert term vanishes identically.
    pa = _plan(_spec(eta=0.0, m=0.48), 10, 100.0, n=200)
    pb = _plan(_spec(eta=0.0, m=0.72), 10, 100.0, n=200)
    assert pb.method == "lambert_ode_m_neq_k"
    np.testing.assert_allclose(pb.quotes, pa.quotes, atol=5e-3)


# ---------------------------------------------------------------------------
# m != k semi-explicit path (paper Thm. 6.1).
# ---------------------------------------------------------------------------


def test_m_neq_k_continuity_in_m() -> None:
    # m -> k from both sides must approach the closed form (Remark 6.2).
    pk = _plan(_spec(), 10, 120.0, n=300)
    p_up = _plan(_spec(m=0.48 + 0.005), 10, 120.0, n=300)
    p_dn = _plan(_spec(m=0.48 - 0.005), 10, 120.0, n=300)
    assert np.nanmax(np.abs(p_up.quotes - pk.quotes)) < 0.2
    assert np.nanmax(np.abs(p_dn.quotes - pk.quotes)) < 0.2


def test_m_gt_k_posts_deeper() -> None:
    # Paper Fig. 9: faster impact decay (m > k) lets the trader post deeper.
    pk = _plan(_spec(), 20, 300.0, n=200)
    pm = _plan(_spec(m=0.60), 20, 300.0, n=200)
    assert pm.quotes[0, 20] > pk.quotes[0, 20]


def test_m_lt_k_posts_tighter() -> None:
    pk = _plan(_spec(), 20, 300.0, n=200)
    pm = _plan(_spec(m=0.40), 20, 300.0, n=200)
    assert pm.quotes[0, 20] < pk.quotes[0, 20]


def test_lambert_domain_violation_fails_closed() -> None:
    # m far below k with large eta pushes the W0 argument below -1/e.
    spec = _spec(m=0.05, eta=5.0)
    with pytest.raises(ValueError, match="Lambert W0"):
        _plan(spec, 20, 300.0, n=50)


def test_plan_rejects_invalid_grids() -> None:
    spec = _spec()
    with pytest.raises(ValueError):
        optimal_execution_plan(spec, 0, 100.0)
    with pytest.raises(ValueError):
        optimal_execution_plan(spec, 5, 0.0)
    with pytest.raises(ValueError):
        optimal_execution_plan(spec, 5, 100.0, n_grid=1)


# ---------------------------------------------------------------------------
# kappa -> infinity reduction to aggressive-only Almgren-Chriss.
# ---------------------------------------------------------------------------


def test_aggressive_schedule_delegates_to_almgren_chriss() -> None:
    spec = _spec()
    got = aggressive_schedule(20_000.0, 10, spec, eta_temp=1e-6, gamma_perm=0.5, risk_aversion=1e-4)
    ref = almgren_chriss_trajectory(
        20_000.0, 10, sigma=spec.sigma, eta=1e-6, gamma=0.5, risk_aversion=1e-4
    )
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-12)


def test_kappa_infinity_recovers_immediate_aggressive_liquidation() -> None:
    # As the mapped urgency kappa -> inf, both views collapse to immediate
    # execution: the AC schedule liquidates everything in the first slice,
    # and the passive quote dives so deep that a fill is certain in any dt.
    spec = _spec()
    ra = 1e8
    sched = aggressive_schedule(20_000.0, 10, spec, eta_temp=1e-6, gamma_perm=0.0, risk_aversion=ra)
    np.testing.assert_allclose(sched[1:], 0.0, atol=1e-8)
    kappa = urgency_kappa(_spec(phi=1e8, sigma=1.0), eta_temp=1e-6)
    assert kappa > 1e6
    # The passive-optimal mirror: phi >> 0 drives delta*(0, q0) deep through
    # the mid, so the passive order is marketable — a certain fill, i.e.
    # exactly the aggressive-only AC behaviour the kappa -> inf limit gives.
    plan = _plan(_spec(phi=100.0), 5, 50.0, n=40)
    d_star = plan.quote_at(0.0, 5)
    assert d_star < 0.0
    assert float(fill_probability(d_star, 0.5, spec.lam, spec.k)) == pytest.approx(1.0, abs=1e-8)


def test_urgency_kappa_closed_form() -> None:
    spec = _spec(phi=4.0, sigma=0.5)
    assert urgency_kappa(spec, 0.25) == pytest.approx(math.sqrt(4.0 * 0.25 / 0.25))


# ---------------------------------------------------------------------------
# Aggressive fallback pricing composes execution.impact.
# ---------------------------------------------------------------------------


def test_aggressive_fill_price_sell_discount() -> None:
    spec = _spec()
    px = aggressive_fill_price(
        100.0, 10_000.0, 1e9, 0.02, spec, upsilon=0.6, gamma_perm=0.5, half_spread_bps=1.0
    )
    assert 0.0 < px < 100.0
    buy = aggressive_fill_price(
        100.0,
        10_000.0,
        1e9,
        0.02,
        spec,
        upsilon=0.6,
        gamma_perm=0.5,
        half_spread_bps=1.0,
        side=1,
    )
    assert buy > 100.0


def test_aggressive_fill_price_monotone_in_size() -> None:
    spec = _spec()
    small = aggressive_fill_price(100.0, 1_000.0, 1e9, 0.02, spec)
    large = aggressive_fill_price(100.0, 500_000.0, 1e9, 0.02, spec)
    assert large < small


# ---------------------------------------------------------------------------
# Fluid path + diagnostics curves.
# ---------------------------------------------------------------------------


def test_expected_inventory_path_monotone_and_depleting() -> None:
    plan = _plan()
    inv = expected_inventory_path(plan)
    assert inv[0] == pytest.approx(20.0)
    assert np.all(np.diff(inv) <= 1e-12)
    assert inv[-1] < 1.0


def test_fill_probability_curve_decreasing() -> None:
    spec = _spec()
    curve = fill_probability_curve(np.linspace(0.0, 15.0, 30), 5.0, spec)
    assert np.all(np.diff(np.asarray(curve)) < 0.0)


# ---------------------------------------------------------------------------
# Simulator: seeded SYNTHETIC paths.
# ---------------------------------------------------------------------------


def _sim(spec: PassiveImpactSpec | None = None, seed: int = 7, **kw: float):
    plan = _plan(spec or _spec(), q0=10, T=120.0, n=120)
    return plan, simulate_passive_execution(
        plan, mid_price=100.0, unit_size=1000.0, n_paths=64, seed=seed, **kw
    )


def test_simulator_runs_and_labels_synthetic() -> None:
    _, sim = _sim()
    assert sim["label"] == "SYNTHETIC"
    assert all(k.startswith("sim_internal_") or k in ("label", "n_paths") for k in sim)
    assert 0.0 < float(sim["sim_internal_fill_fraction_mean"]) <= 1.0
    assert float(sim["sim_internal_trading_time_mean"]) <= 120.0


def test_simulator_determinism_bit_identical() -> None:
    _, a = _sim(seed=11)
    _, b = _sim(seed=11)
    assert a == b


def test_simulator_zero_ofi_and_zero_impact_unaffected_midprice() -> None:
    # eta = beta = sigma = 0: fills cannot move the price — the zero-OFI
    # reduction.  Terminal mid must equal arrival exactly.
    spec = _spec(eta=0.0, beta_ofi=0.0, sigma=0.0)
    _, sim = _sim(spec)
    assert float(sim["sim_internal_terminal_price_mean"]) == pytest.approx(100.0)


def test_simulator_planted_ofi_drift_exact() -> None:
    # sigma = eta = 0 isolates the OFI channel: dS = beta * OFI each step.
    plan = _plan(_spec(eta=0.0, sigma=0.0, beta_ofi=0.01), q0=4, T=60.0, n=60)
    shocks = np.full(60, 7.5)
    sim = simulate_passive_execution(
        plan, mid_price=100.0, unit_size=1000.0, n_paths=8, seed=3, ofi_shocks=shocks
    )
    expected_terminal = 100.0 + 0.01 * float(shocks.sum())
    assert float(sim["sim_internal_terminal_price_mean"]) == pytest.approx(
        expected_terminal, rel=1e-12
    )


def test_simulator_impact_depresses_terminal_price() -> None:
    spec = _spec(sigma=0.0, beta_ofi=0.0)
    _, sim = _sim(spec)
    assert float(sim["sim_internal_terminal_price_mean"]) < 100.0


def test_compare_baseline_is_synthetic_and_deterministic() -> None:
    spec = _spec()
    a = compare_vs_aggressive_baseline(
        spec, 8, 120.0, mid_price=100.0, unit_size=1000.0, n_paths=24, seed=5, n_grid=60
    )
    b = compare_vs_aggressive_baseline(
        spec, 8, 120.0, mid_price=100.0, unit_size=1000.0, n_paths=24, seed=5, n_grid=60
    )
    assert a == b
    assert a["label"] == "SYNTHETIC"
    assert "sim_internal_passive_minus_aggressive_mean" in a


def test_simulator_residual_marked_when_not_liquidating() -> None:
    plan = _plan(_spec(lam=0.05), q0=10, T=60.0, n=60)  # slow fills -> residue
    sim = simulate_passive_execution(
        plan,
        mid_price=100.0,
        unit_size=1000.0,
        n_paths=16,
        seed=2,
        liquidate_residual=False,
    )
    assert float(sim["sim_internal_fill_fraction_mean"]) < 0.6


# ---------------------------------------------------------------------------
# Fail-closed edges.
# ---------------------------------------------------------------------------


def test_spec_validation_fail_closed() -> None:
    with pytest.raises(ValueError):
        PassiveImpactSpec(lam=0.0, k=0.48)
    with pytest.raises(ValueError):
        PassiveImpactSpec(lam=1.0, k=-0.1)
    with pytest.raises(ValueError):
        PassiveImpactSpec(lam=1.0, k=0.48, eta=-1.0)
    with pytest.raises(ValueError):
        PassiveImpactSpec(lam=1.0, k=0.48, sigma=float("nan"))
    with pytest.raises(ValueError):
        PassiveImpactSpec(lam=1.0, k=0.48, tick_size=0.0)


def test_fill_functions_reject_bad_inputs() -> None:
    with pytest.raises(ValueError):
        fill_intensity(np.array([np.inf]), 1.0, 0.5)
    with pytest.raises(ValueError):
        fill_probability(1.0, -1.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        fill_probability(1.0, 1.0, 0.0, 0.5)
    with pytest.raises(ValueError):
        expected_fill_time(np.array([np.nan]), 1.0, 0.5)
    with pytest.raises(ValueError):
        passive_impact_rate(np.array([np.inf]), 1.0, 0.5)


def test_ofi_rejects_negative_flow() -> None:
    with pytest.raises(ValueError):
        order_flow_imbalance(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        ofi_price_response(np.array([np.nan]), 0.1)


def test_quote_at_bounds_and_errors() -> None:
    plan = _plan(_spec(), 6, 60.0, n=10)
    with pytest.raises(ValueError):
        plan.quote_at(0.0, 0)
    with pytest.raises(ValueError):
        plan.quote_at(0.0, 7)
    with pytest.raises(ValueError):
        plan.quote_at(-1.0, 3)
    assert math.isfinite(plan.quote_at(30.0, 3))


def test_value_function_quote_requires_equal_decay() -> None:
    with pytest.raises(ValueError, match="m == k"):
        value_function_quote(0.0, 1, _spec(m=0.6), 5, 50.0)


def test_simulator_rejects_bad_seed_and_shapes() -> None:
    plan = _plan(_spec(), 4, 60.0, n=12)
    with pytest.raises(ValueError):
        simulate_passive_execution(plan, mid_price=100.0, unit_size=1000.0, n_paths=4, seed="x")
    with pytest.raises(ValueError):
        simulate_passive_execution(
            plan,
            mid_price=100.0,
            unit_size=1000.0,
            n_paths=4,
            seed=1,
            ofi_shocks=np.zeros(5),
        )
    with pytest.raises(ValueError):
        simulate_passive_execution(plan, mid_price=-1.0, unit_size=1000.0, n_paths=4, seed=1)


def test_phi_zero_polynomial_limit_is_finite() -> None:
    plan = _plan(_spec(phi=0.0), 5, 50.0, n=40)
    assert np.all(np.isfinite(plan.quotes[:, 1:]))


def test_paper_table1_quote_scale() -> None:
    # SYNTHETIC sanity anchor, not market evidence: for Table-1 parameters
    # the optimal quote at t=0 sits at a few ticks above mid.
    plan = _plan()
    d0 = plan.quote_at(0.0, 20)
    assert 3.0 < d0 < 12.0
