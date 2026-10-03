"""Tests for execution/langevin_impact.py — wave-19 lane W19.

Generalized-Langevin latent-liquidity market-impact model (Itkin 2026,
arXiv:2609.37872). Everything here is **labeled SYNTHETIC** correctness
validation — closed-form equalities (Prop. 2-4, Eqs. 20-52), golden pins
against the paper's Table 1/Table 3 baseline runs, regime diagnostics
(linear / square-root / linear), vol-scaling, duration-independence,
depletion narrowing, the Eq. 35 round-trip nonnegativity identity,
determinism, and fail-closed edges. Never market evidence. Sim-internal
PnL diagnostics stay namespaced ``sim_internal_*`` and are asserted to
never leak into report blobs.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from quant_fund.execution.impact import pow_law_total_impact
from quant_fund.execution.langevin_impact import (
    COUNTERFLOW_KINDS,
    THRESHOLD_SCALINGS,
    LangevinImpactConfig,
    aggregate_counterflow,
    almgren_sqrt_reference,
    back_loaded_shape,
    baseline_config,
    bench_langevin_impact,
    broad_spectrum_config,
    constant_rate_schedule,
    counterflow_rate,
    counterflow_slope,
    counterflow_time_scale,
    cumulative_counterflow,
    duration_free_impact,
    ewma_sigma,
    exponential_threshold_density,
    flat_shape,
    fresh_pool_config,
    fresh_pool_impact_closed_form,
    front_loaded_shape,
    history_effect,
    impact_curve,
    kyle_config,
    pathwise_upper_bound,
    pause_shape,
    piecewise_rate_schedule,
    pool_intensity,
    post_execution_relaxation,
    quadratic_onset_coefficient,
    regime_scan,
    round_trip_cost,
    shaped_rate_schedule,
    simulate_paths,
    single_mode_config,
    small_order_expansion,
    sqrt_band,
    sqrt_regime_bounds,
    stationary_displacement,
    terminal_impact,
    trajectory_rate_schedule,
)
from quant_fund.models.volatility import ewma_variance

DT = 0.01

# Table-3 golden pins (paper, reference units): fresh exponential-pool impact
# of a unit metaorder executed at constant rate over horizon T.
TABLE3_FRESH = {
    0.3: 0.269390,
    1.0: 0.144834,
    3.0: 0.082776,
    10.0: 0.045057,
}
TABLE3_GLE_T1 = 0.158264
TABLE3_SINGLE_T1 = 0.155852


def _times(horizon: float, dt: float = DT) -> np.ndarray:
    return np.arange(0.0, horizon + 0.5 * dt, dt)


# ---------------------------------------------------------------------------
# Shared fixtures (deterministic; the expensive sims are computed once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fresh_paths():
    cfg = fresh_pool_config()
    return simulate_paths(
        cfg,
        constant_rate_schedule(1.0, 1.0),
        _times(1.0),
        n_paths=1,
        exec_horizon=1.0,
        seed=1,
    )


@pytest.fixture(scope="module")
def gle_paths():
    return simulate_paths(
        baseline_config(),
        constant_rate_schedule(1.0, 1.0),
        _times(1.0),
        n_paths=256,
        exec_horizon=1.0,
        seed=7,
    )


@pytest.fixture(scope="module")
def stationary_paths():
    t = np.arange(0.0, 20.0 + 0.5 * DT, DT)
    return simulate_paths(
        baseline_config(),
        lambda _: 0.0,
        t,
        n_paths=256,
        exec_horizon=20.0,
        seed=3,
    )


@pytest.fixture(scope="module")
def round_trip_paths():
    t = _times(2.0)
    rate = piecewise_rate_schedule(np.array([1.0, -1.0]), np.array([0.0, 1.0, 2.0]))
    return simulate_paths(
        baseline_config(),
        rate,
        t,
        n_paths=256,
        exec_horizon=2.0,
        seed=11,
    )


@pytest.fixture(scope="module")
def curve_fresh():
    sizes = np.logspace(-4, 4, 25)
    return impact_curve(fresh_pool_config(), sizes, 1.0, n_paths=1, dt=DT, seed=1)


@pytest.fixture(scope="module")
def curve_gle():
    sizes = np.logspace(-4, 4, 25)
    return impact_curve(baseline_config(), sizes, 1.0, n_paths=128, dt=DT, seed=1)


# ---------------------------------------------------------------------------
# Config fail-closed
# ---------------------------------------------------------------------------


def test_config_rejects_bad_counterflow_kind() -> None:
    with pytest.raises(ValueError, match="counterflow_kind"):
        LangevinImpactConfig(counterflow_kind="power")


def test_config_rejects_bad_threshold_scaling() -> None:
    with pytest.raises(ValueError, match="threshold_scaling"):
        LangevinImpactConfig(threshold_scaling="log")


def test_config_rejects_nonpositive_depth() -> None:
    with pytest.raises(ValueError):
        LangevinImpactConfig(depth=0.0)


def test_config_rejects_mismatched_kernel_lengths() -> None:
    with pytest.raises(ValueError):
        LangevinImpactConfig(intrinsic_weights=(0.5, 0.05), intrinsic_rates=(1.0,))


def test_config_rejects_bad_pool_params() -> None:
    with pytest.raises(ValueError):
        LangevinImpactConfig(pool_y_rho=0.0)
    with pytest.raises(ValueError):
        LangevinImpactConfig(pool_floor=1.0)


def test_config_rejects_negative_sigma_y() -> None:
    with pytest.raises(ValueError):
        LangevinImpactConfig(sigma_y=-0.1)


# ---------------------------------------------------------------------------
# Counterflow / pool primitives
# ---------------------------------------------------------------------------


def test_threshold_density_is_normalized_and_eq25() -> None:
    chi = np.linspace(0.0, 40.0, 20001)
    pi = exponential_threshold_density(chi, q_star=100.0, d_c=1.0)
    assert float(np.trapezoid(pi, chi)) == pytest.approx(100.0, rel=1e-6)
    assert pi[0] == pytest.approx(100.0)
    assert pi[1000] == pytest.approx(100.0 * math.exp(-chi[1000]), rel=1e-12)


def test_aggregate_counterflow_matches_exponential_closed_form() -> None:
    # Eq. 20 quadrature on the Eq. 24 exponential density must reproduce the
    # Eq. 25 closed form counterflow_rate("exponential").
    chi = np.linspace(0.0, 25.0, 20001)
    dens = exponential_threshold_density(chi, q_star=100.0, d_c=1.0)
    d = np.linspace(-3.0, 3.0, 61)
    numeric = aggregate_counterflow(d, chi, dens)
    exact = counterflow_rate(d, kind="exponential", q_star=100.0, d_c=1.0)
    assert np.max(np.abs(numeric - exact)) < 2e-3


def test_counterflow_rate_odd_monotone() -> None:
    d = np.linspace(-5.0, 5.0, 101)
    for kind in COUNTERFLOW_KINDS:
        a = counterflow_rate(d, kind=kind, q_star=100.0, d_c=1.0)
        assert np.allclose(a, -a[::-1], atol=1e-12)
        assert np.all(np.diff(a) >= 0.0)
        assert a[d == 0.0] == 0.0


def test_counterflow_slope_matches_symmetric_difference() -> None:
    d = np.linspace(0.05, 4.0, 60)
    for kind in COUNTERFLOW_KINDS:
        s = counterflow_slope(d, kind=kind, q_star=100.0, d_c=1.0)
        eps = 1e-6
        num = (
            counterflow_rate(d + eps, kind=kind, q_star=100.0, d_c=1.0)
            - counterflow_rate(d - eps, kind=kind, q_star=100.0, d_c=1.0)
        ) / (2.0 * eps)
        assert np.max(np.abs(s - num)) < 1e-5
        assert np.all(s >= 0.0)


def test_quadratic_onset_and_time_scale() -> None:
    # Baseline: omega = q*/(2 d_c^2) = 50, tau_cf = L0/(omega d_c) = 0.02.
    assert quadratic_onset_coefficient(100.0, 1.0) == pytest.approx(50.0)
    assert counterflow_time_scale(50.0, 1.0, 1.0) == pytest.approx(0.02)
    d = np.array([0.1, 1.0, 3.0])
    a = counterflow_rate(d, kind="quadratic", q_star=100.0, d_c=1.0)
    assert np.allclose(a, 50.0 * d**2)


def test_pool_intensity_logistic_identities() -> None:
    y = np.linspace(-6.0, 6.0, 121)
    rho = pool_intensity(y, y_rho=2.0, floor=0.3)
    assert pool_intensity(np.array([0.0]), y_rho=2.0, floor=0.3)[0] == pytest.approx(1.0)
    assert np.allclose(rho + pool_intensity(-y, y_rho=2.0, floor=0.3), 2.0)
    assert np.all(rho >= 0.3) and np.all(rho <= 1.7)
    edge = pool_intensity(np.array([-40.0, 40.0]), y_rho=2.0, floor=0.3)
    assert edge[0] == pytest.approx(1.7, abs=1e-6)
    assert edge[1] == pytest.approx(0.3, abs=1e-6)
    assert np.all(np.diff(rho) < 0.0)  # decreasing in y
    # Disabled pool is identically 1.
    assert np.all(pool_intensity(y, y_rho=2.0, floor=0.3, enabled=False) == 1.0)


# ---------------------------------------------------------------------------
# Rate schedules
# ---------------------------------------------------------------------------


def test_constant_schedule_integrates_to_quantity() -> None:
    r = constant_rate_schedule(2.5, 3.0)
    t = np.linspace(0.0, 3.0, 3001)
    mid = 0.5 * (t[1:] + t[:-1])
    # Midpoint quadrature on the step window [0, T).
    assert float(np.sum(np.vectorize(r)(mid)) * (t[1] - t[0])) == pytest.approx(2.5, rel=1e-9)
    assert r(3.0) == pytest.approx(0.0)


def test_shaped_schedules_preserve_quantity() -> None:
    t = np.linspace(0.0, 2.0, 4001)
    for shape in (
        flat_shape,
        front_loaded_shape(2.0),
        back_loaded_shape(2.0),
        pause_shape(2.0, 0.3),
    ):
        r = shaped_rate_schedule(1.7, 2.0, shape)
        assert float(np.trapezoid(np.vectorize(r)(t), t)) == pytest.approx(1.7, rel=5e-4)


def test_pause_schedule_is_zero_inside_pause() -> None:
    r = shaped_rate_schedule(1.0, 2.0, pause_shape(2.0, 0.4))
    assert r(1.0) == pytest.approx(0.0)
    # Active rate Q/(T(1-kappa)) up to the discrete normalization error.
    assert r(0.5) == pytest.approx(1.0 / (2.0 * (1.0 - 0.4)), rel=1e-3)


def test_piecewise_schedule_rates_and_total() -> None:
    r = piecewise_rate_schedule(np.array([2.0, -1.0]), np.array([0.0, 1.0, 3.0]))
    assert r(0.4) == pytest.approx(2.0)
    assert r(2.0) == pytest.approx(-1.0)
    assert r(3.5) == pytest.approx(0.0)
    t = np.linspace(0.0, 3.0, 3001)
    mid = 0.5 * (t[1:] + t[:-1])
    assert float(np.sum(np.vectorize(r)(mid)) * (t[1] - t[0])) == pytest.approx(0.0, abs=1e-9)


def test_trajectory_schedule_composes_slice_trades() -> None:
    # Holdings 0 -> 1 -> 0.5 -> 0.5 over horizon 3: buys 1.0, sells 0.5, holds.
    r = trajectory_rate_schedule(np.array([0.0, 1.0, 0.5, 0.5]), horizon=3.0)
    t = np.linspace(0.0, 3.0, 3001)
    mid = 0.5 * (t[1:] + t[:-1])
    assert float(np.sum(np.vectorize(r)(mid)) * (t[1] - t[0])) == pytest.approx(0.5, abs=1e-9)
    assert r(0.2) == pytest.approx(1.0)
    assert r(1.5) == pytest.approx(-0.5)
    assert r(2.5) == pytest.approx(0.0)


def test_schedule_fail_closed() -> None:
    with pytest.raises(ValueError):
        constant_rate_schedule(1.0, 0.0)
    with pytest.raises(ValueError):
        constant_rate_schedule(float("nan"), 1.0)
    with pytest.raises(ValueError):
        piecewise_rate_schedule(np.array([1.0, 2.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        trajectory_rate_schedule(np.array([0.5]), horizon=1.0)


# ---------------------------------------------------------------------------
# Closed-form formulas
# ---------------------------------------------------------------------------


def test_fresh_pool_closed_form_tanh_and_limits() -> None:
    # Eq. 45: I = sqrt(Q/(omega T)) tanh(sqrt(omega Q T)/L0).
    assert fresh_pool_impact_closed_form(1.0, 1.0, 50.0, 1.0) == pytest.approx(
        math.sqrt(1.0 / 50.0) * math.tanh(math.sqrt(50.0))
    )
    # Large-Q limit saturates at the pool bound sqrt(Q/(omega T)).
    assert fresh_pool_impact_closed_form(1e4, 1.0, 50.0, 1.0) == pytest.approx(
        math.sqrt(1e4 / 50.0), rel=1e-10
    )
    # Small-Q limit is linear in Q up to the cubic counterflow correction.
    v = fresh_pool_impact_closed_form(1e-4, 1.0, 50.0, 1.0)
    assert v == pytest.approx(1e-4, rel=3e-3)
    assert fresh_pool_impact_closed_form(1e-6, 1.0, 50.0, 1.0) == pytest.approx(1e-6, rel=1e-4)


def test_small_order_expansion_matches_full_form() -> None:
    # Truncation is O((omega Q)^2) relative — tighten the tolerance as Q shrinks.
    for q, tol in ((1e-3, 2e-3), (3e-3, 2e-2), (1e-2, 5e-2)):
        full = fresh_pool_impact_closed_form(q, 1.0, 50.0, 1.0)
        exp_ = small_order_expansion(q, 1.0, 50.0, 1.0)
        assert abs(exp_ - full) / full < tol
    # Expansion keeps the linear leading term and the cubic correction.
    assert small_order_expansion(0.01, 1.0, 50.0, 1.0) == pytest.approx(
        0.01 - 50.0 * 0.01**2 / 3.0, rel=1e-12
    )


def test_stationary_displacement_solves_balance() -> None:
    for kind in COUNTERFLOW_KINDS:
        d_inf = stationary_displacement(2.0, kind=kind, q_star=100.0, d_c=1.0, rho=0.8)
        lhs = 0.8 * float(counterflow_rate(np.array([d_inf]), kind=kind, q_star=100.0, d_c=1.0)[0])
        assert lhs == pytest.approx(2.0, rel=1e-9)
    assert stationary_displacement(0.0, kind="quadratic", q_star=100.0, d_c=1.0) == 0.0


def test_post_execution_relaxation_is_hyperbolic() -> None:
    tau = np.array([0.0, 0.5, 1.0, 5.0])
    out = post_execution_relaxation(0.2, tau, omega=50.0, depth=1.0)
    assert np.allclose(out, 0.2 / (1.0 + 50.0 * 0.2 * tau))
    with pytest.raises(ValueError):
        post_execution_relaxation(0.2, np.array([-1.0]), omega=50.0, depth=1.0)


def test_duration_free_impact_forms() -> None:
    dur = duration_free_impact(
        1.0, sigma=1.0, c_tau=1.0, pi_hat_zero=100.0, depth=1.0, form="duration"
    )
    ela = duration_free_impact(
        1.0, sigma=1.0, c_tau=1.0, pi_hat_zero=100.0, depth=1.0, form="elapsed"
    )
    x = math.sqrt(100.0 / 2.0)
    assert dur == pytest.approx(math.sqrt(2.0 / 100.0) * math.tanh(x))
    from scipy.special import i0, i1

    assert ela == pytest.approx(math.sqrt(2.0 / 100.0) * float(i1(2.0 * x) / i0(2.0 * x)))
    assert 0.0 < ela < dur
    with pytest.raises(ValueError):
        duration_free_impact(1.0, sigma=1.0, c_tau=1.0, pi_hat_zero=100.0, depth=1.0, form="bad")


def test_sqrt_regime_bounds_ordering_and_scaling() -> None:
    lo, hi = sqrt_regime_bounds(50.0, 1.0, 1.0, 1.0)
    assert lo == pytest.approx(1.0 / 50.0)
    assert hi == pytest.approx(50.0)
    assert lo < 1.0 < hi
    # The range ratio grows quadratically with the horizon.
    lo2, hi2 = sqrt_regime_bounds(50.0, 1.0, 4.0, 1.0)
    assert hi2 / lo2 == pytest.approx(16.0 * hi / lo)


def test_pathwise_upper_bound_covers_closed_form() -> None:
    for q in (0.1, 1.0, 10.0):
        bound = pathwise_upper_bound(q, 1.0, 50.0, 1.0, rho_min=0.3)
        # Prop. 5: depletion weakens counterflow, effective omega rho_min * omega.
        assert bound == pytest.approx(fresh_pool_impact_closed_form(q, 1.0, 50.0 * 0.3, 1.0))
        assert bound >= fresh_pool_impact_closed_form(q, 1.0, 50.0, 1.0)


# ---------------------------------------------------------------------------
# Simulator golden pins (paper Table 1 / Table 3 baseline)
# ---------------------------------------------------------------------------


def test_fresh_pool_impact_matches_table3(fresh_paths) -> None:
    assert terminal_impact(fresh_paths)[0] == pytest.approx(TABLE3_FRESH[1.0], abs=2e-3)


def test_fresh_pool_impact_matches_table3_horizons() -> None:
    for horizon, target in TABLE3_FRESH.items():
        dt = min(DT, horizon / 500.0)
        t = np.arange(0.0, horizon + 0.5 * dt, dt)
        p = simulate_paths(
            fresh_pool_config(),
            constant_rate_schedule(1.0, horizon),
            t,
            n_paths=1,
            exec_horizon=horizon,
            seed=1,
        )
        assert terminal_impact(p)[0] == pytest.approx(target, abs=3e-3)


def test_fresh_pool_quadratic_matches_prop4() -> None:
    cfg = fresh_pool_config(counterflow_kind="quadratic")
    t = _times(1.0)
    p = simulate_paths(
        cfg, constant_rate_schedule(1.0, 1.0), t, n_paths=1, exec_horizon=1.0, seed=1
    )
    assert terminal_impact(p)[0] == pytest.approx(
        fresh_pool_impact_closed_form(1.0, 1.0, 50.0, 1.0), abs=2e-4
    )


def test_gle_impact_matches_table3(gle_paths) -> None:
    mean, se = terminal_impact(gle_paths)
    assert abs(mean - TABLE3_GLE_T1) < 4.0 * se + 2e-3
    assert 0.13 < mean < 0.19


def test_single_mode_impact_matches_table3() -> None:
    p = simulate_paths(
        single_mode_config(),
        constant_rate_schedule(1.0, 1.0),
        _times(1.0),
        n_paths=256,
        exec_horizon=1.0,
        seed=7,
    )
    mean, se = terminal_impact(p)
    assert abs(mean - TABLE3_SINGLE_T1) < 4.0 * se + 3e-3


def test_kyle_reduction_is_linear_permanent() -> None:
    p = simulate_paths(
        kyle_config(),
        constant_rate_schedule(1.0, 1.0),
        _times(1.0),
        n_paths=4,
        exec_horizon=1.0,
        seed=7,
    )
    # Counterflow and pool disabled: D is the cumulative volume in depth units.
    assert terminal_impact(p)[0] == pytest.approx(1.0, abs=1e-10)
    assert np.all(np.diff(p.displacement[0]) >= 0.0)


def test_noise_free_latent_displacement_and_pool() -> None:
    cfg = replace(baseline_config(), sigma_y=0.0)
    p = simulate_paths(
        cfg,
        constant_rate_schedule(1.0, 1.0),
        _times(1.0),
        n_paths=1,
        exec_horizon=1.0,
        seed=1,
    )
    # Paper Table 1 noise-free run: Y(T=1) ~= 1.01, rho(Y) ~= 0.83.
    assert p.latent[0, -1] == pytest.approx(1.01, abs=0.03)
    assert p.pool_intensity[0, -1] == pytest.approx(0.83, abs=0.04)


def test_stationary_latent_and_pool_statistics(stationary_paths) -> None:
    burn = 500  # discard the first 5 time units
    y = stationary_paths.latent[:, burn:]
    # Table 1 stationary law: std(Y) ~= 0.46, E[rho(Y)] = 1, std ~= 0.08.
    assert float(y.std()) == pytest.approx(0.46, abs=0.08)
    rho = pool_intensity(y, y_rho=2.0, floor=0.3)
    assert float(rho.mean()) == pytest.approx(1.0, abs=0.03)
    assert float(rho.std()) == pytest.approx(0.08, abs=0.04)
    # Stored pool field is rho(sgn(D) Y): identically 1 with no order flow.
    assert np.all(stationary_paths.pool_intensity == 1.0)


def test_pathwise_upper_bound_holds_in_simulation(gle_paths) -> None:
    bound = pathwise_upper_bound(1.0, 1.0, 50.0, 1.0, rho_min=0.3)
    assert np.all(np.abs(gle_paths.displacement[:, -1]) <= bound + 1e-9)
    assert np.all(gle_paths.displacement[:, -1] <= 1.0 + 1e-9)


def test_sim_internal_diagnostics_namespaced(fresh_paths) -> None:
    keys = [k for k in fresh_paths.sim_internal if not k.startswith("sim_internal_")]
    assert keys == []
    assert fresh_paths.sim_internal["sim_internal_net_traded"] == pytest.approx(1.0)
    assert fresh_paths.sim_internal["sim_internal_exec_price_ratio"] > 0.0


def test_determinism_bit_identical() -> None:
    t = _times(0.5)
    r = constant_rate_schedule(0.5, 0.5)
    a = simulate_paths(baseline_config(), r, t, n_paths=16, exec_horizon=0.5, seed=42)
    b = simulate_paths(baseline_config(), r, t, n_paths=16, exec_horizon=0.5, seed=42)
    for fa, fb in (
        (a.displacement, b.displacement),
        (a.latent, b.latent),
        (a.log_price, b.log_price),
        (a.counterflow, b.counterflow),
    ):
        assert np.array_equal(fa, fb)


def test_simulate_fail_closed() -> None:
    cfg = baseline_config()
    with pytest.raises(ValueError):
        simulate_paths(cfg, constant_rate_schedule(1.0, 1.0), np.array([0.0]), n_paths=1)
    with pytest.raises(ValueError):
        simulate_paths(cfg, constant_rate_schedule(1.0, 1.0), np.array([0.0, 0.5, 0.4]), n_paths=1)
    with pytest.raises(ValueError):
        simulate_paths(cfg, lambda _: float("nan"), _times(0.1), n_paths=1)
    with pytest.raises(ValueError):
        simulate_paths(cfg, constant_rate_schedule(1.0, 1.0), _times(0.1), n_paths=0)


# ---------------------------------------------------------------------------
# Regime diagnostics
# ---------------------------------------------------------------------------


def test_local_exponents_small_mid_large(curve_fresh) -> None:
    e = curve_fresh["exponent"]
    s = curve_fresh["sizes"]
    interior = np.isfinite(e)
    # Small-order linear regime.
    assert e[interior][0] == pytest.approx(1.0, abs=0.15)
    # Intermediate square-root regime near V ~ 1 (L0=1, omega=50).
    mid = np.argmin(np.abs(s[interior] - 1.0))
    assert e[interior][mid] == pytest.approx(0.5, abs=0.08)
    # Large-order linear regime.
    assert e[interior][-1] == pytest.approx(1.0, abs=0.1)


def test_impact_monotone_in_size_and_horizon_decays(curve_fresh) -> None:
    assert np.all(np.diff(curve_fresh["impact"]) > 0.0)
    t1 = simulate_paths(
        fresh_pool_config(),
        constant_rate_schedule(1.0, 1.0),
        _times(1.0),
        n_paths=1,
        exec_horizon=1.0,
        seed=1,
    )
    t3 = simulate_paths(
        fresh_pool_config(),
        constant_rate_schedule(1.0, 3.0),
        _times(3.0),
        n_paths=1,
        exec_horizon=3.0,
        seed=1,
    )
    assert terminal_impact(t3)[0] < terminal_impact(t1)[0]


def test_vol_scaling_doubles_sqrt_impact() -> None:
    cfg = fresh_pool_config(
        counterflow_kind="quadratic", threshold_scaling="vol", counterflow_scale=1.0
    )
    vals = []
    for sigma in (1.0, 2.0):
        p = simulate_paths(
            replace(cfg, sigma_x=sigma),
            constant_rate_schedule(1.0, 1.0),
            _times(1.0),
            n_paths=1,
            exec_horizon=1.0,
            seed=5,
        )
        vals.append(terminal_impact(p)[0])
    assert vals[1] / vals[0] == pytest.approx(2.0, rel=0.02)


def test_duration_independence_both_forms() -> None:
    base = fresh_pool_config(counterflow_kind="quadratic", counterflow_scale=1.0)
    for scaling, form in (("duration", "duration"), ("elapsed", "elapsed")):
        cfg = replace(base, threshold_scaling=scaling)
        impacts = []
        for horizon in (1.0, 4.0):
            p = simulate_paths(
                cfg,
                constant_rate_schedule(1.0, horizon),
                _times(horizon),
                n_paths=1,
                exec_horizon=horizon,
                seed=5,
            )
            impacts.append(terminal_impact(p)[0])
        assert impacts[1] / impacts[0] == pytest.approx(1.0, abs=0.03)
        closed = duration_free_impact(
            1.0, sigma=1.0, c_tau=1.0, pi_hat_zero=100.0, depth=1.0, form=form
        )
        assert impacts[0] == pytest.approx(closed, rel=0.02)


def test_depletion_narrows_sqrt_band(curve_fresh, curve_gle) -> None:
    bf = sqrt_band(curve_fresh["sizes"], curve_fresh["impact"])
    bg = sqrt_band(curve_gle["sizes"], curve_gle["impact"])
    assert bf["found"] == 1.0
    assert bg["width_decades"] < bf["width_decades"]
    # Paper Table 4 direction: fresh band several decades, GLE band ~1.
    assert 1.5 < bf["width_decades"] < 3.5
    assert bg["width_decades"] < 1.2


def test_round_trip_cost_identity_nonnegative(round_trip_paths) -> None:
    out = round_trip_cost(round_trip_paths)
    assert out["rhs_total"] >= 0.0
    # Eq. 35 identity holds up to the martingale term int q dm, whose
    # Monte-Carlo error is O(sigma_X / sqrt(n_paths)): compare within 4 SE.
    p = round_trip_paths
    q = p.rates
    dt = np.diff(p.times)
    cost_i = np.sum(q[None, :] * p.log_price[:, 1 : q.size + 1] * dt[None, :], axis=1)
    rhs_i = 0.5 * p.depth * p.displacement[:, q.size] ** 2 + np.sum(
        p.counterflow[:, 1 : q.size + 1] * p.displacement[:, 1 : q.size + 1] * dt[None, :],
        axis=1,
    )
    resid = cost_i - rhs_i
    se = float(resid.std(ddof=1) / math.sqrt(resid.size))
    assert abs(float(resid.mean())) < 4.0 * se
    assert out["cost"] == pytest.approx(out["rhs_total"], abs=max(4.0 * se, 1e-12))
    assert abs(out["identity_residual"]) < 0.05 * out["rhs_total"]


def test_round_trip_zero_order_is_zero_cost(gle_paths) -> None:
    t = _times(2.0)
    p = simulate_paths(baseline_config(), lambda _: 0.0, t, n_paths=4, exec_horizon=2.0, seed=9)
    out = round_trip_cost(p)
    assert out["rhs_total"] == pytest.approx(0.0, abs=1e-12)


def test_cumulative_counterflow_saturates(fresh_paths) -> None:
    mean, _ = cumulative_counterflow(fresh_paths)
    # Counterflow can never exceed the metaorder quantity on average.
    assert 0.0 < mean < 1.0
    assert mean == pytest.approx(0.84, abs=0.05)


def test_history_effect_fresh_negative_gle_above(fresh_paths, gle_paths) -> None:
    hf = history_effect(
        fresh_pool_config(),
        prior_qty=1.0,
        probe_qty=1.0,
        exec_dur=1.0,
        gap=0.0,
        n_paths=1,
        dt=DT,
        seed=2,
    )
    hg = history_effect(
        baseline_config(),
        prior_qty=1.0,
        probe_qty=1.0,
        exec_dur=1.0,
        gap=0.0,
        n_paths=192,
        dt=DT,
        seed=2,
    )
    assert hf["h"] < 0.0
    assert hg["h"] > hf["h"]
    assert hf["residual_displacement"] > 0.0
    assert hg["j_prior_probe_se"] > 0.0


def test_schedule_front_back_ordering() -> None:
    cfg = fresh_pool_config()
    t = _times(1.0)
    impacts = {}
    for name, shape in (
        ("flat", flat_shape),
        ("front", front_loaded_shape(1.0)),
        ("back", back_loaded_shape(1.0)),
        ("pause", pause_shape(1.0, 0.3)),
    ):
        p = simulate_paths(
            cfg,
            shaped_rate_schedule(1.0, 1.0, shape),
            t,
            n_paths=1,
            exec_horizon=1.0,
            seed=0,
        )
        impacts[name] = terminal_impact(p)[0]
    assert impacts["front"] < impacts["flat"] < impacts["back"]
    assert impacts["flat"] < impacts["pause"]


def test_post_execution_decay_hyperbolic_in_sim() -> None:
    # Quadratic fresh pool: post-trade D relaxes like Eq. 50.
    cfg = fresh_pool_config(counterflow_kind="quadratic")
    t = _times(1.0)
    p = simulate_paths(
        cfg, constant_rate_schedule(1.0, 1.0), t, n_paths=1, exec_horizon=1.0, seed=1
    )
    d_t = float(p.displacement[0, -1])
    tau = np.array([0.5, 1.0, 2.0])
    predicted = post_execution_relaxation(d_t, tau, omega=50.0, depth=1.0)
    assert np.all(np.diff(predicted) < 0.0)
    assert predicted[0] == pytest.approx(d_t / (1.0 + 50.0 * d_t * 0.5))


def test_regime_scan_shape_and_monotone_horizons() -> None:
    sizes = np.array([0.01, 0.03, 0.1, 1.0, 10.0])
    out = regime_scan(fresh_pool_config(), sizes, np.array([1.0, 3.0]), n_paths=1, dt=DT, seed=1)
    imp = np.asarray(out["impacts"])
    assert imp.shape == (2, sizes.size)
    assert np.all(np.diff(imp) > 0.0)  # larger orders cost more
    assert np.all(np.diff(imp, axis=0) < 0.0)  # longer horizons cost less


def test_composed_helpers_match_sources() -> None:
    rng = np.random.default_rng(0)
    r = rng.standard_normal(200) * 0.02
    assert ewma_sigma(r, lam=0.94) == pytest.approx(
        math.sqrt(float(ewma_variance(r, lam=0.94)[-1]))
    )
    assert almgren_sqrt_reference(0.01, 0.5, 0.02) == pytest.approx(
        pow_law_total_impact(0.01, 0.5, 0.02, exponent=0.5)
    )


def test_impact_curve_fail_closed() -> None:
    with pytest.raises(ValueError):
        impact_curve(fresh_pool_config(), np.array([1.0, 2.0]), 1.0)
    with pytest.raises(ValueError):
        impact_curve(fresh_pool_config(), np.array([-1.0, 1.0, 2.0, 3.0]), 1.0)


def test_bench_flat_synthetic_dict() -> None:
    out = bench_langevin_impact(seed=0, n_paths=64, dt=DT)
    assert all(isinstance(v, float) for v in out.values())
    assert all(k.startswith("synthetic_") or k == "runtime_seconds" for k in out)
    assert out["synthetic_fresh_pool_abs_err_t1"] < 2e-3
    assert 0.13 < out["synthetic_gle_impact_t1"] < 0.19
    assert out["synthetic_vol_scaling_ratio"] == pytest.approx(2.0, rel=0.05)
    assert out["synthetic_duration_invariance_ratio"] == pytest.approx(1.0, abs=0.05)
    assert out["synthetic_depletion_narrows_band"] == 1.0
    assert out["synthetic_round_trip_min_cost"] >= 0.0


def test_broad_spectrum_config_preserves_integrated_strengths() -> None:
    base = baseline_config()
    broad = broad_spectrum_config(n_modes=4, breadth=1000.0)

    def s_int(c: LangevinImpactConfig) -> float:
        return sum(a / g for a, g in zip(c.intrinsic_weights, c.intrinsic_rates, strict=True))

    def s_flow(c: LangevinImpactConfig) -> float:
        return sum(a / g for a, g in zip(c.flow_amplitudes, c.flow_rates, strict=True))

    assert s_int(broad) == pytest.approx(s_int(base))
    assert s_flow(broad) == pytest.approx(s_flow(base))
    with pytest.raises(ValueError):
        broad_spectrum_config(n_modes=1)


def test_threshold_scaling_modes_change_scale() -> None:
    from quant_fund.execution.langevin_impact import threshold_scale_at

    cfg = baseline_config()
    for mode in THRESHOLD_SCALINGS:
        v = threshold_scale_at(0.5, replace(cfg, threshold_scaling=mode), 2.0)
        assert v > 0.0
    # Duration scaling grows with the horizon; fixed is constant.
    d1 = threshold_scale_at(0.5, replace(cfg, threshold_scaling="duration"), 1.0)
    d4 = threshold_scale_at(0.5, replace(cfg, threshold_scaling="duration"), 4.0)
    assert d4 / d1 == pytest.approx(2.0)
    f1 = threshold_scale_at(0.5, replace(cfg, threshold_scaling="fixed"), 1.0)
    f4 = threshold_scale_at(0.5, replace(cfg, threshold_scaling="fixed"), 4.0)
    assert f1 == f4
