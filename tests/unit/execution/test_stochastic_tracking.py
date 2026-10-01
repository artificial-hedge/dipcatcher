"""Tests for execution/stochastic_tracking.py — wave 17 lane.

Nutz & Voss (2026, arXiv:2608.29468) generalized Obizhaeva--Wang execution
with a stochastic terminal target and the sharp O(sqrt(eps)) regularization
rate. Everything here is SYNTHETIC correctness material: closed-form
equalities (OW schedule, Hilbert-form cost, Chen--Horst--Tran benchmark),
exponential-kernel convolution exactness, seeded Monte-Carlo rate sweeps,
fail-closed edges, one determinism test. Never market evidence, no Sharpe.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.execution.almgren_chriss import twap_trajectory
from quant_fund.execution.stochastic_tracking import (
    ExplicitStrategy,
    RegularizedSolution,
    TrackingGrid,
    UnregularizedSolution,
    cht_regularized,
    cht_value,
    explicit_strategy,
    hilbert_sq_norm,
    impact_cost,
    impact_process,
    optimal_unregularized,
    ow_benchmark_value,
    ow_coefficients,
    position_trades,
    regularization_gap_sweep,
    regularized_cost,
    sharp_rate_constant,
    sim_delayed_revelation,
    sim_gaussian_revelation,
    sim_instant_revelation,
    tr_seminorm,
    translation_modulus,
)

T, BETA, LAM, XI, N = 1.0, 1.0, 1.0, 1.0, 200
D = BETA * T + 2.0
FORBIDDEN_HEADLINE_TOKENS = ("sharpe", "sortino", "calmar", "nav")


def _grid(n: int = N, horizon: float = T) -> TrackingGrid:
    return TrackingGrid(horizon, n)


def _coeffs(g: TrackingGrid, beta: float = BETA, lam: float = LAM):
    return ow_coefficients(beta, lam, g)


def _sol(g: TrackingGrid, xi: float = XI):
    return optimal_unregularized(xi, _coeffs(g), g)


def _all_keys(obj: object) -> list[str]:
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(str(k))
            keys.extend(_all_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys.extend(_all_keys(item))
    return keys


# ---------------------------------------------------------------------------
# Grid / coefficients
# ---------------------------------------------------------------------------


def test_grid_times_and_validation() -> None:
    g = _grid(10)
    assert g.times.shape == (11,)
    assert g.times[0] == 0.0 and g.times[-1] == pytest.approx(T)
    assert g.step == pytest.approx(T / 10)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            TrackingGrid(bad, 10)
    for bad_n in (0, 1, -3, 2.5, True):
        with pytest.raises(ValueError):
            TrackingGrid(1.0, bad_n)  # type: ignore[arg-type]


def test_coefficients_constant_closed_form() -> None:
    g = _grid()
    c = _coeffs(g)
    # eta = 1/2, theta_t = beta t/(2 lam), alpha = 1/(2 lam), weight = beta/lam
    assert np.all(c.eta == pytest.approx(0.5))
    assert c.theta[-1] == pytest.approx(BETA * T / (2 * LAM))
    assert c.theta_dot[-1] == pytest.approx(BETA / (2 * LAM))
    assert c.alpha[0] == pytest.approx(1.0 / (2 * LAM))
    assert c.weight[0] == pytest.approx(BETA / LAM)
    assert np.all(c.gamma_dot == 0.0)


def test_coefficients_time_varying_lam() -> None:
    g = _grid(50)
    t = g.times
    lam_t = 1.0 + 0.3 * t  # lam_dot = 0.3 constant
    c = ow_coefficients(BETA, lam_t, g)
    assert np.all(np.isfinite(c.theta))
    # gamma_dot_i = lam_dot/lam_i for i >= 1
    assert c.gamma_dot[10] == pytest.approx(0.3 / lam_t[10], rel=1e-10)
    # eta = (beta + gamma_dot)/(2 beta + gamma_dot) per eq. (4.2)
    i = 7
    expected_eta = (BETA + c.gamma_dot[i]) / (2 * BETA + c.gamma_dot[i])
    assert c.eta[i] == pytest.approx(expected_eta)


def test_coefficients_failclosed() -> None:
    g = _grid()
    with pytest.raises(ValueError):
        ow_coefficients(-0.5, LAM, g)  # negative resilience
    with pytest.raises(ValueError):
        ow_coefficients(BETA, 0.0, g)  # zero depth
    with pytest.raises(ValueError):
        ow_coefficients(BETA, -1.0, g)
    # lam decaying so fast that 2 beta + gamma_dot hits zero -> price
    # manipulation admitted, Assumption 4.1(iii) violated
    t = g.times
    with pytest.raises(ValueError):
        ow_coefficients(BETA, np.exp(-10.0 * t), g)
    with pytest.raises(ValueError):
        ow_coefficients(np.array([BETA, BETA]), LAM, g)  # wrong length
    with pytest.raises(ValueError):
        ow_coefficients(BETA, np.full(g.n_steps + 1, np.nan), g)


# ---------------------------------------------------------------------------
# Impact kernel: exact discrete exponential convolution (eq. 4.3)
# ---------------------------------------------------------------------------


def test_impact_single_trade_exponential_decay() -> None:
    g = _grid(100)
    c = _coeffs(g)
    trades = np.zeros(101)
    trades[40] = 2.0
    y = impact_process(trades, c, g)
    assert y[39] == 0.0
    assert y[40] == pytest.approx(LAM * 2.0)
    expected = LAM * 2.0 * np.exp(-BETA * (g.times - g.times[40]))
    assert np.allclose(y[40:], expected[40:], atol=1e-12)


def test_impact_convolution_matches_brute_force() -> None:
    g = _grid(80)
    c = ow_coefficients(2.0, 1.5, g)
    rng = np.random.default_rng(7)
    trades = rng.standard_normal(81)
    y = impact_process(trades, c, g, y0=0.4)
    # B_i = sum_{k=1..i} beta_k h with B_0 = 0 (trade at t_0 undecayed)
    b_int = np.concatenate([[0.0], np.cumsum(np.full(80, 2.0 * g.step))])
    brute = np.array(
        [
            0.4 * math.exp(-b_int[i])
            + sum(math.exp(-(b_int[i] - b_int[j])) * 1.5 * trades[j] for j in range(i + 1))
            for i in range(81)
        ]
    )
    assert np.allclose(y, brute, atol=1e-12)


def test_impact_batch_and_pathwise_consistency() -> None:
    g = _grid(50)
    c = _coeffs(g)
    rng = np.random.default_rng(11)
    batch = rng.standard_normal((9, 51))
    y = impact_process(batch, c, g)
    assert y.shape == (9, 51)
    for p in range(9):
        assert np.allclose(y[p], impact_process(batch[p], c, g))


def test_impact_failclosed() -> None:
    g = _grid(10)
    c = _coeffs(g)
    with pytest.raises(ValueError):
        impact_process(np.zeros(7), c, g)  # wrong length
    with pytest.raises(ValueError):
        impact_process(np.full(11, np.nan), c, g)
    with pytest.raises(ValueError):
        impact_process(np.zeros(11), c, g, y0=np.inf)


# ---------------------------------------------------------------------------
# Unregularized optimum (Theorem 4.2) — deterministic benchmark
# ---------------------------------------------------------------------------


def test_optimal_deterministic_ow_schedule() -> None:
    g = _grid()
    sol = _sol(g)
    t = g.times
    # interior: Q^0_t = Xi (1 + beta t)/D exactly (eq. 4.10 / Prop. 5.9)
    expected = XI * (1.0 + BETA * t[:N]) / D
    assert np.allclose(sol.positions[0, :N], expected, atol=1e-12)
    assert sol.positions[0, N] == XI  # Q_T = Xi_T exactly


def test_optimal_blocks_eq_4_11() -> None:
    g = _grid()
    sol = _sol(g)
    # initial and terminal blocks are Xi/D each (Prop. 5.9)
    assert sol.block_initial[0] == pytest.approx(XI / D)
    assert sol.block_terminal[0] == pytest.approx(XI / D)


def test_optimal_impact_equals_eta_martingale() -> None:
    g = _grid()
    sol = _sol(g)
    # eq. (4.9): Y_t = eta_t M_t on [0, T) up to grid resolution of trade cells
    interior = sol.impact[0, :N]
    expected = 0.5 * sol.martingale[0, :N]
    assert np.abs(interior - expected).max() < 2 * g.step


def test_optimal_value_closed_form() -> None:
    g = _grid()
    sol = _sol(g)
    assert sol.value == pytest.approx(LAM * XI**2 / D, rel=1e-10)
    assert sol.value == pytest.approx(ow_benchmark_value(XI, BETA, LAM, T))


def test_impact_cost_of_optimal_matches_value() -> None:
    g = _grid()
    sol = _sol(g)
    c = _coeffs(g)
    j0 = float(impact_cost(sol.positions, c, g)[0])
    # Hilbert-form cost of Q^0 equals the analytic V(0) up to grid quadrature
    assert j0 == pytest.approx(sol.value, rel=1e-3)


def test_optimal_is_minimizer_against_perturbation() -> None:
    g = _grid()
    sol = _sol(g)
    c = _coeffs(g)
    rng = np.random.default_rng(3)
    q = sol.positions.copy()
    q[:, 1:N] += 0.05 * rng.standard_normal((1, N - 1))
    q[:, -1] = XI
    assert float(impact_cost(q, c, g)[0]) > float(impact_cost(sol.positions, c, g)[0])


def test_hilbert_excess_cost_identity() -> None:
    """Lemma 5.3: J_0(Q) - J_0(Q^0) = ||Y^Q - Y^0||_H^2 >= 0.

    Exact in continuous time; on the grid the perturbation identity holds up
    to O(h) because Q^0 solves the continuous problem, not the discrete
    quadratic form -- so assert the residual shrinks with h.
    """
    rng = np.random.default_rng(5)
    residuals: list[float] = []
    for n in (100, 200, 400):
        g = _grid(n)
        c = _coeffs(g)
        sol = _sol(g)
        q = sol.positions.copy()
        q[:, 5 : n // 2] += 0.1 * rng.standard_normal((1, n // 2 - 5))
        q[:, -1] = XI
        z = impact_process(position_trades(q), c, g) - sol.impact
        gap = float(impact_cost(q, c, g)[0] - impact_cost(sol.positions, c, g)[0])
        norm = float(hilbert_sq_norm(z, c, g)[0])
        assert gap > 0.0 and norm > 0.0
        residuals.append(abs(gap - norm) / norm)
    assert residuals[-1] < 0.02
    assert residuals[0] > residuals[-1]  # O(h) convergence of the identity


# ---------------------------------------------------------------------------
# Reductions to repo's AC/OW machinery
# ---------------------------------------------------------------------------


def test_reduction_twap_limit_large_beta() -> None:
    """beta -> inf (instant resilience) recovers TWAP remaining inventory."""
    g = _grid(200)
    sol = optimal_unregularized(XI, ow_coefficients(1e4, LAM, g), g)
    remaining = XI - sol.positions[0]
    twap = twap_trajectory(XI, 200)
    # blocks vanish like 1/(beta T + 2); the schedule is then linear
    assert np.abs(remaining - twap).max() < 0.02


def test_reduction_small_beta_is_two_blocks() -> None:
    """beta -> 0 collapses the schedule to equal t=0 and t=T blocks."""
    g = _grid(100)
    b = 1e-3
    sol = optimal_unregularized(XI, ow_coefficients(b, LAM, g), g)
    d = b * T + 2.0
    assert sol.block_initial[0] == pytest.approx(XI / d)
    assert sol.block_terminal[0] == pytest.approx(XI / d)
    interior_trades = np.diff(sol.positions[0, : g.n_steps])
    assert np.abs(interior_trades).max() < 5e-4


# ---------------------------------------------------------------------------
# Explicit regularized strategy (Lemma 5.5, eqs. 5.18-5.20)
# ---------------------------------------------------------------------------


def test_explicit_strategy_is_ac_and_terminal() -> None:
    g = _grid()
    c = _coeffs(g)
    sol = _sol(g)
    s = explicit_strategy(sol, 0.01, c, g)
    assert isinstance(s, ExplicitStrategy)
    assert s.positions[0, 0] == 0.0  # absolutely continuous: no initial block
    assert s.positions[0, -1] == pytest.approx(XI)  # Q_T = Xi_T
    assert np.all(np.isfinite(s.rates))
    # terminal bridge pins the target: u_i = (Xi - Q_i)/(T - t_i) feedback
    i = s.switch_index + 2
    t_i = g.times[i]
    expected_rate = (XI - s.positions[0, i]) / (T - t_i)
    assert s.rates[0, i] == pytest.approx(expected_rate, rel=1e-2, abs=1e-6)


def test_explicit_strategy_linear_bridge_deterministic() -> None:
    g = _grid()
    c = _coeffs(g)
    sol = _sol(g)
    eps = 0.04
    a = math.sqrt(eps)
    s = explicit_strategy(sol, eps, c, g)
    k = s.switch_index
    q_start = s.positions[0, k]
    # deterministic Xi -> bridge is the straight segment (T-t)/a * Qbar + ...
    interior = np.arange(k + 1, N)
    lin = q_start + (XI - q_start) * (g.times[interior] - g.times[k]) / a
    assert np.allclose(s.positions[0, interior], lin, atol=1e-10)


def test_explicit_filter_matches_closed_form() -> None:
    g = _grid()
    c = _coeffs(g)
    sol = _sol(g)
    eps = 0.01
    a = math.sqrt(eps)
    s = explicit_strategy(sol, eps, c, g)
    # Qbar_i = (1/a) sum_{j<=i} Q^0_{j-1} int_{t_{j-1}}^{t_j} e^{-(t_i-r)/a} dr
    q0 = sol.positions[0]
    f = math.exp(-g.step / a)
    for i in (1, s.switch_index // 2, s.switch_index):
        brute = sum(
            q0[j - 1] * (math.exp(-(g.times[i] - g.times[j]) / a) * (1.0 - f))
            for j in range(1, i + 1)
        )
        assert s.positions[0, i] == pytest.approx(brute)


def test_explicit_strategy_failclosed_eps() -> None:
    g = _grid()
    c = _coeffs(g)
    sol = _sol(g)
    for bad in (0.0, -1e-3, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            explicit_strategy(sol, bad, c, g)
    with pytest.raises(ValueError):
        explicit_strategy(sol, T * T / 4.0 + 1e-9, c, g)


def test_explicit_strategy_monotone_tracking_in_eps() -> None:
    g = _grid()
    c = _coeffs(g)
    sol = _sol(g)
    eps_grid = np.array([3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    tl = [explicit_strategy(sol, float(e), c, g).tracking_l2 for e in eps_grid]
    # tracking L2 shrinks as eps -> 0 (bounded by C sqrt(eps) in the paper)
    assert np.all(np.diff(tl) > 0.0)


def test_q_eps_l2_converges_to_q0() -> None:
    g = _grid(400)
    c = _coeffs(g)
    sol = _sol(g)
    for e in (0.05, 0.01, 0.004):
        s = explicit_strategy(sol, e, c, g)
        assert s.tracking_l2 < 0.05


# ---------------------------------------------------------------------------
# Chen--Horst--Tran benchmark + sharp sqrt(eps) rate (Prop. 5.9)
# ---------------------------------------------------------------------------


def test_cht_optimizer_terminal_and_shape() -> None:
    g = _grid()
    r = cht_regularized(XI, BETA, LAM, 0.01, g)
    assert isinstance(r, RegularizedSolution)
    assert r.positions[-1] == pytest.approx(XI)
    # AC minimizer: no initial block (a_eps - c_eps sinh K_eps = 0 identically)
    assert r.positions[0] == 0.0
    # rate is the grid derivative of the position where smooth (interior cells)
    mid = slice(g.n_steps // 4, 3 * g.n_steps // 4)
    num_rate = np.diff(r.positions)[mid] / g.step
    assert np.allclose(num_rate, r.rates[1:-1][mid], rtol=5e-2, atol=1e-3)


def test_cht_value_monotone_and_limit() -> None:
    eps = np.array([0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-7])
    vals = np.array([cht_value(e, XI, BETA, LAM, T) for e in eps])
    v0 = ow_benchmark_value(XI, BETA, LAM, T)
    assert np.all(vals >= v0)
    assert np.all(np.diff(vals) < 0.0)
    assert vals[-1] == pytest.approx(v0, rel=1e-3)


def test_cht_value_equals_regularized_objective() -> None:
    g = _grid(400)
    c = _coeffs(g)
    eps = 0.01
    r = cht_regularized(XI, BETA, LAM, eps, g)
    numeric = float(regularized_cost(r.positions, eps, c, g)[0])
    assert numeric == pytest.approx(r.value, rel=5e-3)


def test_sqrt_eps_rate_cht_slope() -> None:
    """Prop. 5.9: V(eps) - V(0) ~ C sqrt(eps); fitted log-log slope ~ 0.5."""
    g = _grid(200)
    eps = np.logspace(-6, -2, 9)
    sweep = regularization_gap_sweep(eps, XI, BETA, LAM, T, g.n_steps)
    slope = sweep["synthetic_loglog_slope_cht"]
    assert 0.45 <= float(slope) <= 0.55


def test_sharp_constant_matches_expansion() -> None:
    """(V(eps)-V(0))/sqrt(eps) -> 2 sqrt(beta lam) Xi^2/(beta T+2)^2."""
    g = _grid(200)
    c_star = sharp_rate_constant(XI, BETA, LAM, T)
    sweep = regularization_gap_sweep(np.array([1e-6, 3e-6, 1e-5]), XI, BETA, LAM, T, g.n_steps)
    ratios = np.asarray(sweep["synthetic_value_gap_cht"]) / np.sqrt(np.asarray(sweep["eps"]))
    assert np.allclose(ratios, c_star, rtol=0.05)
    assert c_star == pytest.approx(2.0 * np.sqrt(BETA * LAM) * XI**2 / D**2)


def test_explicit_strategy_excess_sqrt_rate() -> None:
    """Thm 5.7: J_0(Q^eps) - J_0(Q^0) = O(sqrt(eps)); slope in [0.4, 0.6]."""
    g = _grid(400)
    # keep sqrt(eps) >= ~5h so the grid resolution floor does not bias the fit
    h = g.step
    eps = np.geomspace((5 * h) ** 2, 0.04, 8)
    sweep = regularization_gap_sweep(eps, XI, BETA, LAM, T, g.n_steps)
    slope = float(sweep["synthetic_loglog_slope_explicit"])
    assert 0.4 <= slope <= 0.6
    assert np.all(np.asarray(sweep["synthetic_excess_explicit"]) >= 0.0)


def test_tracking_l2_sqrt_rate() -> None:
    g = _grid(400)
    h = g.step
    eps = np.geomspace((5 * h) ** 2, 0.04, 8)
    sweep = regularization_gap_sweep(eps, XI, BETA, LAM, T, g.n_steps)
    slope = float(sweep["synthetic_loglog_slope_tracking"])
    assert 0.4 <= slope <= 0.65


def test_explicit_strategy_cost_dominates_optimizer_value() -> None:
    """V(eps) is the inf over AC strategies: J_eps(Q^eps) >= V(eps)."""
    g = _grid(400)
    c = _coeffs(g)
    sol = _sol(g)
    for eps in (0.04, 0.01, 0.0025):
        s = explicit_strategy(sol, eps, c, g)
        j_eps = float(regularized_cost(s.positions, eps, c, g)[0])
        v_eps = cht_value(eps, XI, BETA, LAM, T)
        assert j_eps >= v_eps - 2e-3


# ---------------------------------------------------------------------------
# Stochastic terminal target (random Xi_T via martingale paths)
# ---------------------------------------------------------------------------


def test_instant_revelation_value_closed_form() -> None:
    """Deterministic-schedule-per-path => V(0) = lam E[Xi_T^2]/D."""
    g = _grid(100)
    c = _coeffs(g)
    xi = sim_instant_revelation(4000, g, 0.5, 1.0, rng=0)
    sol = optimal_unregularized(xi, c, g)
    analytic = LAM * float(np.mean(xi[:, -1] ** 2)) / D
    assert sol.value == pytest.approx(analytic, rel=1e-10)


def test_delayed_revelation_martingale_structure() -> None:
    g = _grid(100)
    c = _coeffs(g)
    r = N // 4
    xi = sim_delayed_revelation(64, g, 0.5, 1.0, r, rng=1)
    sol = optimal_unregularized(xi, c, g)
    m = sol.martingale
    # M is constant before the jump and jumps once at t_r
    assert np.allclose(np.diff(m[:, :r], axis=1), 0.0)
    assert np.allclose(np.diff(m[:, r + 1 :], axis=1), 0.0)
    # jump size = w_r * dXi_r with w_r = lam_T/(1 + lam_T (theta_T - theta_r))
    w_r = LAM / (1.0 + LAM * (c.theta[-1] - c.theta[r]))
    d_xi = xi[:, r] - xi[:, r - 1]
    assert np.allclose(m[:, r] - m[:, r - 1], w_r * d_xi)
    # terminal constraint exactly on every path
    assert np.allclose(sol.positions[:, -1], xi[:, -1])


def test_gaussian_revelation_is_martingale_and_terminal() -> None:
    g = _grid(100)
    xi = sim_gaussian_revelation(3000, g, 0.5, 1.0, last_info_index=60, rng=2)
    # pathwise: constant after information stops; martingale: constant mean
    assert np.allclose(xi[:, 61:], xi[:, 60][:, np.newaxis])
    assert float(np.mean(xi[:, -1])) == pytest.approx(0.5, abs=0.06)
    sol = optimal_unregularized(xi, _coeffs(g), g)
    assert np.allclose(sol.positions[:, -1], xi[:, -1])


def test_stochastic_terminal_reached_by_explicit_strategy() -> None:
    g = _grid(100)
    c = _coeffs(g)
    xi = sim_delayed_revelation(128, g, 0.5, 1.0, N // 4, rng=3)
    sol = optimal_unregularized(xi, c, g)
    s = explicit_strategy(sol, 0.01, c, g)
    assert np.allclose(s.positions[:, -1], xi[:, -1])


def test_stochastic_excess_sqrt_rate_seeded() -> None:
    """Seeded MC: mean J_0(Q^eps) - J_0(Q^0) stays O(sqrt(eps))."""
    n = 120
    g = TrackingGrid(T, n)
    h = g.step
    eps = np.geomspace((6 * h) ** 2, 0.04, 7)
    rng_xi = 4
    xi = sim_delayed_revelation(256, g, 0.5, 0.8, n // 3, rng=rng_xi)
    sweep = regularization_gap_sweep(eps, xi, BETA, LAM, T, n)
    slope = float(sweep["synthetic_loglog_slope_explicit"])
    assert 0.35 <= slope <= 0.7
    assert np.all(np.asarray(sweep["synthetic_excess_explicit"]) >= -1e-12)


def test_stochastic_value_at_least_deterministic_mean() -> None:
    """Random terminal target never beats the deterministic cost: E[Xi^2] >= Xi^2."""
    g = _grid(100)
    xi = sim_instant_revelation(2048, g, 0.5, 1.0, rng=5)
    sol = optimal_unregularized(xi, _coeffs(g), g)
    assert sol.value > ow_benchmark_value(0.5, BETA, LAM, T)


# ---------------------------------------------------------------------------
# Time-translation modulus (eqs. 2.6-2.7)
# ---------------------------------------------------------------------------


def test_translation_modulus_linear_path_exact() -> None:
    """For x_t = c t, omega(h) = c^2 h^2 (T - h) + c^2 h^3/3."""
    g = _grid(50)
    c_slope = 2.0
    x = c_slope * g.times[np.newaxis, :]
    lags = np.array([1, 2, 4])
    omega = translation_modulus(x, lags, g)
    h = g.step
    for i, k in enumerate(lags):
        lag = k * h
        expected = c_slope**2 * (lag**2 * (T - lag) + lag**3 / 3.0)
        # grid discretization: sum over cells approximates the integrals
        assert omega[i] == pytest.approx(expected, rel=5e-2)


def test_tr_seminorm_finite_for_q0() -> None:
    """Q^0 has a finite translation seminorm (eq. 5.15) -- semimartingale class."""
    g = _grid(100)
    sol = _sol(g)
    sem = tr_seminorm(sol.positions, g)
    assert np.isfinite(sem) and sem > 0.0
    # omega(h) is roughly linear in h -> omega(h)/h roughly constant
    omega = translation_modulus(sol.positions, np.array([2, 10, 40]), g)
    ratios = omega / (np.array([2, 10, 40]) * g.step)
    assert float(np.max(ratios) / np.min(ratios)) < 10.0


def test_translation_modulus_failclosed() -> None:
    g = _grid(10)
    with pytest.raises(ValueError):
        translation_modulus(np.zeros(11), np.array([0]), g)
    with pytest.raises(ValueError):
        translation_modulus(np.zeros(11), np.array([99]), g)
    with pytest.raises(ValueError):
        translation_modulus(np.zeros(11), np.array([1.5]), g)


# ---------------------------------------------------------------------------
# Determinism, honesty, and remaining fail-closed edges
# ---------------------------------------------------------------------------


def test_determinism_bit_identical() -> None:
    g = _grid(80)
    c = _coeffs(g)
    xi = sim_gaussian_revelation(64, g, 0.5, 1.0, 50, rng=9)
    s1 = optimal_unregularized(xi, c, g)
    s2 = optimal_unregularized(xi, c, g)
    assert np.array_equal(s1.positions, s2.positions)
    e1 = explicit_strategy(s1, 0.01, c, g)
    e2 = explicit_strategy(s2, 0.01, c, g)
    assert np.array_equal(e1.positions, e2.positions)
    xi2 = sim_gaussian_revelation(64, g, 0.5, 1.0, 50, rng=9)
    assert np.array_equal(xi, xi2)


def test_honesty_keys_no_forbidden_tokens() -> None:
    sweep = regularization_gap_sweep(np.geomspace(1e-4, 1e-2, 4), XI, BETA, LAM, T, 100)
    keys = _all_keys(sweep)
    for k in keys:
        low = k.lower()
        assert not any(tok in low for tok in FORBIDDEN_HEADLINE_TOKENS)
        assert "pnl" not in low or low.startswith("sim_internal_")


def test_positions_and_cost_failclosed() -> None:
    g = _grid(10)
    c = _coeffs(g)
    with pytest.raises(ValueError):
        impact_cost(np.zeros((3, 7)), c, g)
    with pytest.raises(ValueError):
        impact_cost(np.full(11, np.inf), c, g)
    with pytest.raises(ValueError):
        regularized_cost(np.zeros(11), -0.1, c, g)
    with pytest.raises(ValueError):
        optimal_unregularized(np.zeros(5), c, g)  # wrong length xi
    with pytest.raises(ValueError):
        optimal_unregularized(np.full(11, np.nan), c, g)


def test_simulators_failclosed_and_seeded() -> None:
    g = _grid(10)
    with pytest.raises(ValueError):
        sim_instant_revelation(8, g, 0.0, 1.0, rng=None)
    with pytest.raises(ValueError):
        sim_delayed_revelation(8, g, 0.0, 1.0, 99, rng=0)
    with pytest.raises(ValueError):
        sim_gaussian_revelation(8, g, 0.0, 1.0, 0, rng=0)
    with pytest.raises(ValueError):
        sim_instant_revelation(0, g, 0.0, 1.0, rng=0)
    with pytest.raises(ValueError):
        sim_instant_revelation(8, g, 0.0, -1.0, rng=0)


def test_sweep_failclosed() -> None:
    with pytest.raises(ValueError):
        regularization_gap_sweep(np.array([0.01]), XI, BETA, LAM, T, 50)
    with pytest.raises(ValueError):
        regularization_gap_sweep(np.array([-1.0, 0.01]), XI, BETA, LAM, T, 50)
    with pytest.raises(ValueError):
        regularization_gap_sweep(np.array([0.01, 0.3]), XI, BETA, LAM, T, 50)


def test_solution_types() -> None:
    g = _grid(50)
    sol = _sol(g)
    assert isinstance(sol, UnregularizedSolution)
    assert sol.positions.shape == (1, 51)
    assert sol.martingale.shape == (1, 51)
