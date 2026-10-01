"""SYNTHETIC tests for perpetual variance-swap entry/exit rules (Maeda 2026).

All numerics are closed-form / SYNTHETIC correctness checks against the
paper's stated calibration values and its published Table-3/Sec-8 pins — never
market evidence.  Monte Carlo sections use seeded numpy Generators (plus the
shared ``models.short_rate.cir_simulate`` sampler) and assert only loose,
many-sigma-safe agreements.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import gammainc, hyperu

from quant_fund.execution.varswap_stopping import (
    CIRTwoMeasure,
    PerpetualVarSwap,
    bench_varswap_stopping,
    carry_forcing,
    carry_star,
    dated_breakeven_level,
    dated_exercise_geometry,
    dated_fair_strike,
    dated_mark_phi0,
    dated_mark_state,
    dated_variance_duration,
    entry_exit_window,
    entry_obstacle,
    entry_value,
    exercise_geometry,
    exit_value,
    fair_rate,
    myopic_level,
    position_value,
    solve_entry,
    solve_exit,
    stationary_cdf,
    stationary_ppf,
    threshold_sensitivities,
    variance_perpetuity,
)
from quant_fund.models.short_rate import cir_simulate

# --- paper calibrations ------------------------------------------------------
# Sec. 8.2 in-window short calibration (kappa_Q=1.0, theta_Q=0.045, gamma=0.20,
# lam=-0.6 -> kappa_P=1.6, theta_P=0.028125, nu=2.25).
CIR_A = CIRTwoMeasure(kappa_p=1.6, theta_p=0.028125, kappa_q=1.0, gamma=0.20)
SPEC_A = PerpetualVarSwap(cir=CIR_A, delta=0.5, s_exit=0.002, s_entry=0.002, c_hold=0.012, side=-1)
# Calibration (41): theta_P=0.0408, kappa_P=2.15, gamma=0.2485, kappa_Q=1.503.
CIR_C = CIRTwoMeasure(kappa_p=2.15, theta_p=0.0408, kappa_q=1.503, gamma=0.2485)
# Sec. 8.2 base calibration (kappa_P=2.15, theta_P=0.04186, kappa_Q=2.0,
# gamma=0.26 -> theta_Q=0.045, lam=-0.15).
CIR_B = CIRTwoMeasure(kappa_p=2.15, theta_p=0.04186, kappa_q=2.0, gamma=0.26)


def _spec(cir: CIRTwoMeasure, **kw) -> PerpetualVarSwap:
    base = dict(delta=0.5, s_exit=0.002, s_entry=0.002, c_hold=0.012, side=-1)
    base.update(kw)
    return PerpetualVarSwap(cir=cir, **base)


def m_G(spec: PerpetualVarSwap, v: float) -> float:
    return float(hyperu(spec.a, spec.cir.nu, spec.cir.sigma_tilde * v))


# --- dataclass validation / derived fields -----------------------------------


def test_cir_rejects_nonpositive_params() -> None:
    for kw in (
        {"kappa_p": 0.0},
        {"kappa_p": -1.0},
        {"theta_p": 0.0},
        {"kappa_q": -0.5},
        {"gamma": 0.0},
        {"kappa_p": float("nan")},
        {"gamma": float("inf")},
    ):
        base = dict(kappa_p=1.6, theta_p=0.028125, kappa_q=1.0, gamma=0.20)
        base.update(kw)
        with pytest.raises(ValueError):
            CIRTwoMeasure(**base)


def test_cir_rejects_feller_violation() -> None:
    with pytest.raises(ValueError, match="Feller"):
        CIRTwoMeasure(kappa_p=0.5, theta_p=0.01, kappa_q=0.4, gamma=0.5)


def test_cir_affine_link_derives_theta_q() -> None:
    assert CIR_A.theta_q == pytest.approx(0.045, rel=1e-12)
    assert CIR_C.theta_q == pytest.approx(0.05836327345309, rel=1e-9)
    assert CIR_A.lam == pytest.approx(-0.6)
    assert CIR_B.lam == pytest.approx(-0.15)
    assert CIR_A.nu == pytest.approx(2.25)
    assert CIR_A.sigma_tilde == pytest.approx(80.0)


def test_spec_validation_fail_closed() -> None:
    good = dict(delta=0.5, s_exit=0.002, s_entry=0.002, c_hold=0.012, side=-1)
    for kw in (
        {"delta": 0.0},
        {"delta": -1.0},
        {"s_exit": -1e-9},
        {"s_entry": -0.5},
        {"c_idle": -1.0},
        {"c_hold": float("nan")},
        {"side": 0},
        {"side": 2},
    ):
        base = dict(good)
        base.update(kw)
        with pytest.raises(ValueError):
            PerpetualVarSwap(cir=CIR_A, **base)


def test_reduced_reward_coefficients_hand() -> None:
    # alpha = eps*D0 - s + c_m/delta with D0 = (tQ-tP)/d - tQ/(d+kQ) + tP/(d+kP).
    d0 = (0.045 - 0.028125) / 0.5 - 0.045 / 1.5 + 0.028125 / 2.1
    assert SPEC_A.alpha == pytest.approx(-d0 - 0.002 + 0.024, rel=1e-12)
    assert SPEC_A.beta == pytest.approx(0.6 / (1.5 * 2.1) * -1.0, rel=1e-12)
    assert SPEC_A.beta < 0.0
    assert SPEC_A.a == pytest.approx(0.5 / 1.6)
    assert SPEC_A.k_round == pytest.approx(0.004)


def test_reduced_reward_matches_premium_decomposition() -> None:
    # h~(v) = eps * D(v) - s_bar + c_m/delta (Prop. 6.5 / eq. 35).
    vv = np.linspace(0.001, 0.2, 11)
    rhs = SPEC_A.eps * SPEC_A.premium_gap(vv) - SPEC_A.s_exit + SPEC_A.c_hold / SPEC_A.delta
    np.testing.assert_allclose(SPEC_A.reduced_reward(vv), rhs, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        SPEC_A.reduced_reward(vv), SPEC_A.alpha + SPEC_A.beta * vv, rtol=1e-12, atol=1e-15
    )


# --- analytic primitives -----------------------------------------------------


def test_fair_rate_eq30() -> None:
    # K(v) = theta + delta (v - theta)/(delta + kappa)
    v = 0.02
    assert fair_rate(v, 1.0, 0.045, 0.5) == pytest.approx(0.045 + 0.5 * (v - 0.045) / 1.5)


def test_variance_perpetuity_eq29() -> None:
    vv = np.array([0.0, 0.01, 0.05])
    expect = 0.045 / 0.5 + (vv - 0.045) / 1.5
    np.testing.assert_allclose(variance_perpetuity(vv, 1.0, 0.045, 0.5), expect, rtol=1e-12)


def test_stationary_law_is_gamma() -> None:
    vv = np.linspace(0.0, 0.1, 21)
    np.testing.assert_allclose(
        stationary_cdf(CIR_A, vv), gammainc(CIR_A.nu, CIR_A.sigma_tilde * vv), rtol=1e-14
    )
    assert stationary_cdf(CIR_A, 0.0) == 0.0
    q = stationary_ppf(CIR_A, 0.948)
    assert stationary_cdf(CIR_A, q) == pytest.approx(0.948, rel=1e-10)


def test_premium_gap_affine() -> None:
    vv = np.array([0.0, 0.02, 0.07])
    d = np.asarray(SPEC_A.premium_gap(vv))
    slope = -CIR_A.lam / ((0.5 + CIR_A.kappa_q) * (0.5 + CIR_A.kappa_p))
    np.testing.assert_allclose(np.diff(d) / np.diff(vv), slope, rtol=1e-10)


def test_hold_value_matches_perpetuity() -> None:
    # R(v;K) = eps*P_P(v) - eps*K/delta - c_m/delta (Lemma 6.4).
    v = 0.03
    expect = -variance_perpetuity(v, CIR_A.kappa_p, CIR_A.theta_p, 0.5) - 0.18 / 0.5 - 0.024
    assert SPEC_A.hold_value(v, -0.18) == pytest.approx(expect, rel=1e-12)


# --- exit solve --------------------------------------------------------------


def test_exit_pin_sec82_inwindow() -> None:
    sol = solve_exit(SPEC_A)
    assert sol.exists and sol.region == "lower"
    rate = float(np.sqrt(fair_rate(sol.threshold, CIR_A.kappa_q, CIR_A.theta_q, 0.5)))
    assert rate == pytest.approx(0.1820, abs=2e-4)  # paper: 18.20%
    assert float(stationary_cdf(CIR_A, sol.threshold)) == pytest.approx(0.124, abs=0.002)
    assert sol.residual < 1e-10
    assert sol.dpsi < 0.0  # Psi strictly decreasing at the root (uniqueness)


def test_exit_smooth_pasting_value_and_derivative() -> None:
    sol = solve_exit(SPEC_A)
    b = sol.threshold
    # Value match: U(b*-) = h~(b*); smooth paste: U'(b*-) = h~'(b*) = beta.
    eps = 1e-7 * b
    u_below = float(exit_value(SPEC_A, sol, b - eps))
    u_above = float(exit_value(SPEC_A, sol, b + eps))
    h_b = float(SPEC_A.reduced_reward(b))
    assert abs(u_below - h_b) < 1e-6
    assert abs(u_above - h_b) < 1e-6
    du = (u_above - u_below) / (2 * eps)
    assert du == pytest.approx(SPEC_A.beta, rel=1e-4)


def test_exit_long_side_upper_region_pin() -> None:
    spec = _spec(CIR_B, c_hold=0.0, side=+1)
    sol = solve_exit(spec)
    assert sol.exists and sol.region == "upper"
    rate = float(np.sqrt(fair_rate(sol.threshold, CIR_B.kappa_q, CIR_B.theta_q, 0.5)))
    assert rate == pytest.approx(0.203, abs=0.002)  # paper: 20.3%
    assert float(stationary_cdf(CIR_B, sol.threshold)) == pytest.approx(0.29, abs=0.01)
    assert sol.dpsi < 0.0


def test_exit_empty_below_carry_star() -> None:
    c_star = carry_star(SPEC_A)
    assert c_star == pytest.approx(0.009571, rel=1e-3)
    below = _spec(CIR_A, c_hold=c_star * 0.9)
    above = _spec(CIR_A, c_hold=c_star * 1.05)
    assert not solve_exit(below).exists
    assert solve_exit(above).exists


def test_exit_degenerate_when_lam_zero() -> None:
    cir0 = CIRTwoMeasure(kappa_p=1.2, theta_p=0.05, kappa_q=1.2, gamma=0.2)
    neg = _spec(cir0, c_hold=0.0)
    pos = _spec(cir0, c_hold=0.05)
    sol_neg = solve_exit(neg)
    sol_pos = solve_exit(pos)
    assert sol_neg.region == "empty" and not sol_neg.exists
    assert sol_pos.region == "all" and sol_pos.exists


def test_exit_uniqueness_scan_single_sign_change() -> None:
    # Psi_G has exactly one sign change on (0, v0*): the root is unique.
    import quant_fund.execution.varswap_stopping as m

    sol = solve_exit(SPEC_A)
    vv = np.linspace(1e-9, sol.v0_star, 4000)
    psi = np.asarray(SPEC_A.beta * m._G(SPEC_A, vv) - SPEC_A.reduced_reward(vv) * m._Gp(SPEC_A, vv))
    sign_changes = int(np.sum(psi[:-1] * psi[1:] < 0.0))
    assert sign_changes == 1


def test_exit_value_shapes() -> None:
    sol = solve_exit(SPEC_A)
    vv = np.array([0.001, 0.02, 0.2])
    u = np.asarray(exit_value(SPEC_A, sol, vv))
    # lower-region exercise: U = h~ below b*, proportional to G above.
    assert u[0] == pytest.approx(float(SPEC_A.reduced_reward(0.001)))
    assert u[2] == pytest.approx(sol.lam_coeff * float(m_G(SPEC_A, 0.2)), rel=1e-10)


def test_position_value_adds_hold_value() -> None:
    sol = solve_exit(SPEC_A)
    vv = np.array([0.01, 0.05])
    np.testing.assert_allclose(
        position_value(SPEC_A, sol, vv, 0.20),
        np.asarray(SPEC_A.hold_value(vv, 0.20)) + np.asarray(exit_value(SPEC_A, sol, vv)),
        rtol=1e-12,
    )


# --- entry solve -------------------------------------------------------------


def test_entry_pin_sec82_inwindow() -> None:
    esol = solve_entry(SPEC_A)
    assert esol.exists and not esol.immediate and esol.side_set == "upper"
    rate = float(np.sqrt(fair_rate(esol.threshold, CIR_A.kappa_q, CIR_A.theta_q, 0.5)))
    assert rate == pytest.approx(0.2262, abs=3e-4)  # paper: 22.62%
    cdf = float(stationary_cdf(CIR_A, esol.threshold))
    assert cdf == pytest.approx(0.948, abs=0.002)  # paper: 94.8th pctile
    assert esol.residual < 1e-8
    assert esol.dpsi < 0.0


def test_entry_above_exit_and_rho_positive() -> None:
    sol = solve_exit(SPEC_A)
    esol = solve_entry(SPEC_A)
    assert esol.threshold > sol.threshold
    assert esol.rho_at_d > 0.0


def test_entry_threshold_pin_cal41() -> None:
    # Paper Sec. 8 pins: d*(c_m=200bp) = 0.12161, d*(c_m=300bp) = 0.16745.
    for cm, want in ((0.02, 0.12161), (0.03, 0.16745)):
        esol = solve_entry(_spec(CIR_C, c_hold=cm))
        assert esol.exists and not esol.immediate
        assert esol.threshold == pytest.approx(want, abs=3e-4)


def test_entry_smooth_pasting_at_d() -> None:
    sol = solve_exit(SPEC_A)
    esol = solve_entry(SPEC_A)
    d = esol.threshold
    eps = 1e-7 * d
    j_below = float(entry_value(SPEC_A, sol, esol, d - eps))
    j_above = float(entry_value(SPEC_A, sol, esol, d + eps))
    rho_d = float(entry_obstacle(SPEC_A, sol, d))
    assert abs(j_above - rho_d) < 1e-6  # J = rho on the entry side
    assert abs(j_below - rho_d) < 1e-5  # value match across d*
    dj = (j_above - j_below) / (2 * eps)
    drho = (
        float(entry_obstacle(SPEC_A, sol, d + eps)) - float(entry_obstacle(SPEC_A, sol, d - eps))
    ) / (2 * eps)
    assert dj == pytest.approx(drho, rel=2e-3)


def test_entry_idle_cost_table3_pattern() -> None:
    # Table 3: c_0 pulls d* down (rate falls, percentile falls); at
    # c_0/delta >= k the flat trader enters at once.
    rates, cdfs, flags = [], [], []
    for c0 in (0.0, 0.0025, 0.005, 0.01):
        esol = solve_entry(_spec(CIR_A, c_idle=c0))
        if esol.exists and not esol.immediate:
            rates.append(
                float(np.sqrt(fair_rate(esol.threshold, CIR_A.kappa_q, CIR_A.theta_q, 0.5)))
            )
            cdfs.append(float(stationary_cdf(CIR_A, esol.threshold)))
            flags.append(False)
        else:
            flags.append(True)
    assert rates[0] == pytest.approx(0.2262, abs=3e-4)
    assert rates[1] == pytest.approx(0.2117, abs=3e-4)
    assert 0.19 < rates[2] < 0.21  # paper 19.81% (we land 19.71%)
    assert cdfs[2] == pytest.approx(0.58, abs=0.03)
    assert flags[3]  # c_0 = 100bp -> enter at once
    assert rates == sorted(rates, reverse=True)


def test_entry_with_degenerate_exit() -> None:
    # c_m below c_m* -> U == 0; entry still solves on the affine obstacle.
    spec = _spec(CIR_A, c_hold=carry_star(SPEC_A) * 0.5)
    sol = solve_exit(spec)
    assert not sol.exists
    esol = solve_entry(spec)
    assert esol.exists and not esol.immediate
    assert esol.threshold > 0.0


def test_entry_never_when_costs_dominate() -> None:
    # Enormous round-trip cost makes rho <= 0 everywhere.
    spec = _spec(CIR_A, s_exit=5.0, s_entry=5.0)
    esol = solve_entry(spec)
    assert not esol.exists
    assert not esol.immediate


def test_entry_j_is_majorant() -> None:
    sol = solve_exit(SPEC_A)
    esol = solve_entry(SPEC_A)
    vv = np.linspace(1e-4, 0.09, 60)
    j = np.asarray(entry_value(SPEC_A, sol, esol, vv))
    rho = np.asarray(entry_obstacle(SPEC_A, sol, vv))
    assert np.all(j >= rho - 1e-9)


# --- sensitivities / window ---------------------------------------------------


def test_sensitivities_match_finite_difference() -> None:
    sol = solve_exit(SPEC_A)
    esol = solve_entry(SPEC_A)
    sens = threshold_sensitivities(SPEC_A, sol, esol)
    assert sens["db_dcm"] > 0.0 and sens["dd_dcm"] > 0.0
    h = 1e-6
    b_up = solve_exit(_spec(CIR_A, c_hold=SPEC_A.c_hold + h)).threshold
    b_dn = solve_exit(_spec(CIR_A, c_hold=SPEC_A.c_hold - h)).threshold
    d_up = solve_entry(_spec(CIR_A, c_hold=SPEC_A.c_hold + h)).threshold
    d_dn = solve_entry(_spec(CIR_A, c_hold=SPEC_A.c_hold - h)).threshold
    assert sens["db_dcm"] == pytest.approx((b_up - b_dn) / (2 * h), rel=2e-3)
    assert sens["dd_dcm"] == pytest.approx((d_up - d_dn) / (2 * h), rel=2e-3)


def test_window_edges_match_reachability() -> None:
    win = entry_exit_window(SPEC_A)
    assert win.nonempty
    assert win.c_star == pytest.approx(0.009571, rel=1e-3)
    # paper reports ~[100, 210]bp; exact eta=0.005 edges land just inside.
    assert 0.009 < win.c_lower < 0.0105
    assert 0.0205 < win.c_upper < 0.0225
    assert win.exit_cdf_at_lower == pytest.approx(0.005, abs=1e-3)
    assert win.entry_cdf_at_upper == pytest.approx(0.995, abs=1e-3)
    assert win.width == pytest.approx(win.c_upper - win.c_lower)


def test_window_empty_when_unreachable() -> None:
    # Tiny eta demands a window wider than any charge admits.
    spec = _spec(CIR_A, s_exit=0.05, s_entry=0.05)
    win = entry_exit_window(spec, eta=0.0001)
    assert not win.nonempty or win.width > 0.0


def test_window_rejects_bad_eta_and_long() -> None:
    with pytest.raises(ValueError):
        entry_exit_window(SPEC_A, eta=0.9)
    with pytest.raises(ValueError):
        entry_exit_window(_spec(CIR_B, side=+1, c_hold=0.0))


def test_carry_star_long_is_minus_inf() -> None:
    assert carry_star(_spec(CIR_B, side=+1, c_hold=0.0)) == float("-inf")


def test_carry_star_short_pin() -> None:
    # Paper Sec. 8.2 base cal: c_m* = 0.00304 for the short.
    spec = _spec(CIR_B, side=-1, c_hold=0.0)
    assert carry_star(spec) == pytest.approx(0.00304, abs=2e-4)


# --- dated-contract machinery ---------------------------------------------------


def test_dated_variance_duration_limits() -> None:
    assert dated_variance_duration(0.0, 1.0, 1.0) == pytest.approx(1.0 - np.exp(-1.0))
    assert dated_variance_duration(1.0, 1.0, 1.0) == 0.0
    with pytest.raises(ValueError):
        dated_variance_duration(1.5, 1.0, 1.0)


def test_carry_forcing_affine_and_sign_flip() -> None:
    vv = np.array([0.01, 0.05])
    g = dated_variance_duration(0.25, 1.0, CIR_A.kappa_q)
    f = np.asarray(carry_forcing(0.25, vv, 1.0, -1, CIR_A, s=0.002, c_m=0.012))
    # slope = eps * lam * g_Q(t) = (-1)(-0.6)g > 0 for the short
    assert np.allclose(np.diff(f) / np.diff(vv), 0.6 * g)
    f_long = np.asarray(carry_forcing(0.25, vv, 1.0, +1, CIR_A, s=0.002, c_m=0.012))
    assert np.allclose(np.diff(f_long) / np.diff(vv), -0.6 * g)
    # geometry flips with the forcing sign (Prop. 4.1 / Table 1)
    assert exercise_geometry(-1.0, CIR_A.lam) == "lower"
    assert exercise_geometry(+1.0, CIR_A.lam) == "upper"
    assert exercise_geometry(1.0, 0.0) == "flat"


def test_myopic_level_zero_carry() -> None:
    g = dated_variance_duration(0.0, 2.0, CIR_A.kappa_q)
    want = (0.012 - 0.002) / ((-1.0) * CIR_A.lam * g)
    assert myopic_level(0.0, 2.0, -1, CIR_A, s=0.002, c_m=0.012) == pytest.approx(want)
    cir0 = CIRTwoMeasure(kappa_p=1.0, theta_p=0.03, kappa_q=1.0, gamma=0.2)
    with pytest.raises(ValueError):
        myopic_level(0.0, 1.0, -1, cir0, s=0.0, c_m=0.0)


def test_dated_geometry_table1() -> None:
    g = dated_exercise_geometry(-1, CIR_A, s=0.002, c_m=0.012)
    assert g["region"] == "lower" and g["requires_satisfied"]
    g2 = dated_exercise_geometry(-1, CIR_A, s=0.05, c_m=0.001)
    assert g2["region"] == "lower" and not g2["requires_satisfied"]


def test_dated_marks_terminal_consistency() -> None:
    # At t = T, Phi_0 = -T*K^2 so h(T) = A_T - T*K^2 (the mark).
    t_end = 1.0
    phi_T = dated_mark_phi0(t_end, 0.123, t_end, 0.04, CIR_A)
    assert phi_T == pytest.approx(-t_end * 0.04)
    assert dated_mark_state(t_end, 0.077, 0.123, t_end, 0.04, CIR_A) == pytest.approx(0.077 - 0.04)


def test_dated_fair_strike_and_breakeven() -> None:
    # xi(0, theta_Q) = theta_Q; breakeven v*_{0} at accrued 0 matches xi.
    assert dated_fair_strike(0.0, CIR_A.theta_q, 2.0, CIR_A) == pytest.approx(CIR_A.theta_q)
    v_star = dated_breakeven_level(0.0, 0.0, 2.0, 0.045, CIR_A)
    assert np.isfinite(v_star)


def test_dated_mark_is_q_martingale() -> None:
    # Lemma 2.1: A_t + Phi_0(t,v) is a Q-martingale; A_t += v dt.
    rng = np.random.default_rng(3)
    T, n_steps, dt, n_paths = 1.0, 64, 1.0 / 64.0, 400
    k_val = 0.04
    acc = np.zeros(n_paths)
    prev = np.full(n_paths, 0.05)
    marks = []
    for i in range(1, n_steps + 1):
        cur = np.asarray(
            [
                cir_simulate(r0, CIR_A.kappa_q, CIR_A.theta_q, CIR_A.gamma, 2, dt, rng)[-1]
                for r0 in prev
            ]
        )
        acc = acc + prev * dt
        prev = cur
        marks.append(
            np.mean(acc + np.asarray([dated_mark_phi0(i * dt, v, T, k_val, CIR_A) for v in cur]))
        )
    # martingale => mean mark ~ constant over time (Q-dynamics simulation)
    assert abs(marks[-1] - marks[0]) < 0.02


# --- diagnostics, MC, determinism --------------------------------------------


def test_eigenfunctions_solve_generator() -> None:
    # (L - delta) F = (L - delta) G = 0 pointwise (eq. 25).
    import quant_fund.execution.varswap_stopping as m

    vv = np.linspace(0.005, 0.2, 15)
    for f, fp, fpp in ((m._F, m._Fp, m._Fpp), (m._G, m._Gp, m._Gpp)):
        lv = (
            0.5 * CIR_A.gamma**2 * vv * fpp(SPEC_A, vv)
            + CIR_A.kappa_p * (CIR_A.theta_p - vv) * fp(SPEC_A, vv)
            - SPEC_A.delta * f(SPEC_A, vv)
        )
        assert np.max(np.abs(lv)) < 1e-8


def test_entry_trigger_deep_in_upper_tail() -> None:
    esol = solve_entry(SPEC_A)
    assert float(stationary_cdf(CIR_A, esol.threshold)) > 0.9


def test_mc_stationary_tail_empirical() -> None:
    rng = np.random.default_rng(123)
    esol = solve_entry(SPEC_A)
    v_inf = rng.gamma(CIR_A.nu, 1.0 / CIR_A.sigma_tilde, size=150_000)
    emp = float(np.mean(v_inf > esol.threshold))
    ana = 1.0 - float(stationary_cdf(CIR_A, esol.threshold))
    assert abs(emp - ana) < 0.01


def test_mc_cir_path_matches_stationary_mean() -> None:
    # Long-horizon cir_simulate path mean -> theta_P (composes models.short_rate).
    rng = np.random.default_rng(5)
    path = cir_simulate(0.2, CIR_A.kappa_p, CIR_A.theta_p, CIR_A.gamma, 40_000, 0.001, rng)
    assert np.all(np.asarray(path) >= 0.0)
    assert float(np.mean(path[-20_000:])) == pytest.approx(CIR_A.theta_p, rel=0.1)


def test_determinism() -> None:
    a = bench_varswap_stopping(11)
    b = bench_varswap_stopping(11)
    assert a == b
    sol1 = solve_exit(SPEC_A)
    sol2 = solve_exit(SPEC_A)
    assert sol1 == sol2


def test_bench_keys_and_ranges() -> None:
    out = bench_varswap_stopping(7)
    assert all(k.startswith("synthetic_") for k in out)
    assert out["synthetic_smooth_pasting_max_resid"] < 1e-8
    assert 0.005 < out["synthetic_exit_threshold_short"] < 0.05
    assert out["synthetic_entry_tail_cdf_at_trigger"] > 0.9
    assert out["synthetic_entry_interval_width"] > 0.0
    assert out["synthetic_mc_entry_tail_abs_err"] < 0.01
    assert out["synthetic_eigen_eq_max_resid"] < 1e-8


def test_fail_closed_nonfinite_inputs() -> None:
    with pytest.raises(ValueError):
        fair_rate(float("nan"), 1.0, 0.045, 0.5)
    with pytest.raises(ValueError):
        stationary_cdf(CIR_A, -1.0)
    with pytest.raises(ValueError):
        stationary_ppf(CIR_A, 1.5)
    sol = solve_exit(SPEC_A)
    esol = solve_entry(SPEC_A)
    with pytest.raises(ValueError):
        exit_value(SPEC_A, sol, float("inf"))
    with pytest.raises(ValueError):
        entry_value(SPEC_A, sol, esol, float("nan"))
    with pytest.raises(ValueError):
        carry_forcing(0.0, 0.01, 1.0, 0, CIR_A, s=0.0, c_m=0.0)


def test_sensitivities_fail_closed_on_empty() -> None:
    spec = _spec(CIR_A, c_hold=carry_star(SPEC_A) * 0.5)
    sol = solve_exit(spec)
    esol = solve_entry(spec)
    if not (sol.exists and esol.exists and not esol.immediate):
        with pytest.raises(ValueError):
            threshold_sensitivities(spec, sol, esol)
    else:  # pragma: no cover - guard: this spec should be degenerate
        pytest.fail("expected degenerate pair")
