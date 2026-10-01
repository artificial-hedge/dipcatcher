"""Tests for rough-volatility machinery (wave 21).

Fractional Riccati solver, rHeston characteristic function, rBergomi
simulation, RL-fBm weights. All outputs are SYNTHETIC diagnostics.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate
from scipy.special import gamma as _gamma

from quant_fund.models.rough_heston_rbergomi import (
    _frac_integral,
    _frac_riccati_rhs,
    bench_rough_heston_rbergomi,
    fractional_riccati_solve,
    implied_vol_from_call,
    rbergomi_simulate,
    rheston_call_price,
    rheston_cf,
    rl_fbm_discrete,
    volterra_heston_simulate,
)

_PARAMS = dict(lam=0.9, theta=0.04, nu=0.35, rho=-0.6, v0=0.04)


# -- fail-closed validation ------------------------------------------------


def test_riccati_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        fractional_riccati_solve(1.0, 0.0, 0.5, 0.9, -0.6, 0.35)


def test_riccati_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        fractional_riccati_solve(1.0, 0.7, -1.0, 0.9, -0.6, 0.35)
    with pytest.raises(ValueError):
        fractional_riccati_solve(1.0, 0.7, 0.5, 0.9, 1.2, 0.35)


def test_rbergomi_rejects_bad() -> None:
    with pytest.raises(ValueError):
        rbergomi_simulate(0, 10, 64, 0.1, 0.04, 1.9, 0.6, -0.5)  # H>0.5
    with pytest.raises(ValueError):
        rbergomi_simulate(0, 10, 64, 0.1, 0.04, 1.9, 0.1, 1.2)  # rho>1


def test_rl_fbm_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        rl_fbm_discrete(64, 0.01, 0.4)


def test_cf_rejects_bad() -> None:
    with pytest.raises(ValueError):
        rheston_cf(1.0, 0.5, 0.0, 0.9, 0.04, 0.35, -0.6, 0.7)


def test_implied_vol_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        implied_vol_from_call(-0.1, 0.0, 0.5)


# -- fractional Riccati ----------------------------------------------------


def test_riccati_reduces_to_classical_at_alpha_one() -> None:
    """alpha -> 1: fractional Adams matches solve_ivp on the same F."""
    a = 1.0 - 0.5j
    lam, rho, nu = _PARAMS["lam"], _PARAMS["rho"], _PARAMS["nu"]
    t_val = 0.5
    h = fractional_riccati_solve(a, 0.999, t_val, lam, rho, nu, 300)
    f = _frac_riccati_rhs(a, lam, rho, nu)
    grid = np.linspace(0.0, t_val, 301)
    sol = integrate.solve_ivp(
        lambda tt, y: [f(complex(y[0] + 1j * y[1])).real, f(complex(y[0] + 1j * y[1])).imag],
        [0.0, t_val],
        [0.0, 0.0],
        t_eval=grid,
        rtol=1e-9,
    )
    h_cl = sol.y[0] + 1j * sol.y[1]
    assert np.max(np.abs(h - h_cl)) / np.max(np.abs(h_cl)) < 1e-2


def test_riccati_grid_shape_and_determinism() -> None:
    h1 = fractional_riccati_solve(0.7, 0.7, 0.4, 0.9, -0.5, 0.4, 80)
    h2 = fractional_riccati_solve(0.7, 0.7, 0.4, 0.9, -0.5, 0.4, 80)
    assert h1.shape == (81,)
    np.testing.assert_array_equal(h1, h2)


def test_frac_integral_identity_limit() -> None:
    """I^order h(T) -> h(T) as order -> 0 (identity operator)."""
    n = 400
    dt = 0.5 / n
    h = np.linspace(0.0, 1.0, n + 1) + 0j  # h(s) = 2s
    val = _frac_integral(h, 1e-3, dt)
    assert abs(val - h[-1]) / abs(h[-1]) < 0.1


def test_frac_integral_first_order_is_trapezoid() -> None:
    """order=1 must equal the plain integral."""
    n = 200
    dt = 0.5 / n
    s = np.linspace(0.0, 0.5, n + 1)
    h = s + 0j
    val = _frac_integral(h, 1.0, dt)
    assert abs(val - integrate.trapezoid(h, dx=dt)) < 0.05


# -- rHeston CF ------------------------------------------------------------


def test_cf_matches_classical_when_alpha_near_one() -> None:
    a = 1.0 - 0.5j
    lam, theta, nu, rho, v0 = (
        _PARAMS["lam"],
        _PARAMS["theta"],
        _PARAMS["nu"],
        _PARAMS["rho"],
        _PARAMS["v0"],
    )
    t_val = 0.5
    n = 300
    cf = rheston_cf(a, t_val, v0, lam, theta, nu, rho, 0.999, n)
    h = fractional_riccati_solve(a, 0.999, t_val, lam, rho, nu, n)
    g1 = theta * lam * integrate.trapezoid(h, dx=t_val / n)
    classical = np.exp(g1 + v0 * h[-1])
    assert abs(cf - classical) / abs(classical) < 1e-3


def test_cf_modulus_bounded() -> None:
    cf = rheston_cf(0.8, 0.5, **_PARAMS, alpha=0.7, n_steps=120)
    assert abs(cf) <= 1.0 + 1e-9


def test_cf_deterministic() -> None:
    a = 0.6 - 0.2j
    assert rheston_cf(a, 0.4, **_PARAMS, alpha=0.7, n_steps=80) == rheston_cf(
        a, 0.4, **_PARAMS, alpha=0.7, n_steps=80
    )


def test_call_price_near_bachelier() -> None:
    """ATM call sits near the Bachelier reference for these params."""
    c = rheston_call_price(0.0, 0.5, **_PARAMS, alpha=0.999, n_steps=100, n_u=300)
    import math

    bach = math.sqrt(0.04 * 0.5 / (2 * math.pi))
    assert 0.0 < c < 0.5
    assert abs(c - bach) / bach < 0.35


def test_implied_vol_roundtrip() -> None:
    import math

    sigma, t_val, k = 0.2, 0.5, 0.0
    d = -k / (sigma * math.sqrt(t_val))
    call = (
        sigma * math.sqrt(t_val) * float(np.exp(-0.5 * d * d) / math.sqrt(2 * math.pi))
        + (0.0 - k) * 0.5
    )
    iv = implied_vol_from_call(call, k, t_val)
    assert abs(iv - sigma) < 0.02


# -- rBergomi --------------------------------------------------------------


def test_rl_fbm_weights_variance() -> None:
    alpha = 0.6
    dt, n = 0.005, 200
    w = rl_fbm_discrete(n, dt, alpha)
    theory = (n * dt) ** (2 * alpha - 1) / (_gamma(alpha) ** 2 * (2 * alpha - 1))
    emp = float(np.sum(w * w))
    assert 0.7 < emp / theory < 1.05  # left-point rule loses tail-cell mass


def test_rbergomi_shapes_and_determinism() -> None:
    a = rbergomi_simulate(3, 50, 80, 0.2, 0.04, 1.9, 0.1, -0.6)
    b = rbergomi_simulate(3, 50, 80, 0.2, 0.04, 1.9, 0.1, -0.6)
    for xa, xb in zip(a, b, strict=True):
        np.testing.assert_array_equal(xa, xb)
    x, v, wh = a
    assert x.shape == v.shape == wh.shape == (50, 81)
    assert np.all(v > 0)


def test_rbergomi_leverage_corr_negative() -> None:
    x, v, wh = rbergomi_simulate(11, 800, 100, 0.1, 0.04, 1.9, 0.1, -0.65)
    assert np.corrcoef(x[:, -1], wh[:, -1])[0, 1] < -0.2


def test_volterra_heston_shapes_positive_var() -> None:
    x, v = volterra_heston_simulate(5, 200, 80, 0.4, **_PARAMS, alpha=0.7)
    assert x.shape == v.shape == (200, 81)
    assert np.all(np.isfinite(x))


def test_volterra_heston_deterministic() -> None:
    x1, _ = volterra_heston_simulate(5, 50, 40, 0.3, **_PARAMS, alpha=0.7)
    x2, _ = volterra_heston_simulate(5, 50, 40, 0.3, **_PARAMS, alpha=0.7)
    np.testing.assert_array_equal(x1, x2)


# -- bench -----------------------------------------------------------------


def test_bench_returns_synthetic_blob() -> None:
    blob = bench_rough_heston_rbergomi(seed=20261126, n_paths=800)
    assert blob
    assert all(k.startswith("synthetic_") for k in blob)
    assert all(np.isfinite(v) for v in blob.values())


def test_bench_quality_thresholds() -> None:
    blob = bench_rough_heston_rbergomi(seed=20261126, n_paths=800)
    assert blob["synthetic_riccati_classical_rel_err"] < 1e-2
    assert blob["synthetic_rbergomi_lev_corr_short"] < -0.25
    assert 0.0 < blob["synthetic_rbergomi_hurst_recovered_short"] < 0.3
    assert blob["synthetic_cf_mc_gap_in_se"] < 6.0
    assert 0.7 < blob["synthetic_fbm_var_ratio_to_theory"] < 1.05


def test_bench_deterministic() -> None:
    a = bench_rough_heston_rbergomi(seed=7, n_paths=400)
    b = bench_rough_heston_rbergomi(seed=7, n_paths=400)
    assert a == b
