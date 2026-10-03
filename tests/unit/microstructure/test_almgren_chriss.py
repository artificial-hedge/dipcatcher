"""Tests for microstructure/almgren_chriss.py — AC frontier, transient
propagator, MC consistency. SYNTHETIC drills only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.microstructure.almgren_chriss import (
    ALMGREN_CHRISS_SCHEMA,
    ACParams,
    ac_bench,
    efficient_frontier,
    frontier_point,
    optimal_trajectory,
    trade_list,
)


def test_params_fail_closed() -> None:
    with pytest.raises(ValueError):
        ACParams(X=0.0)
    with pytest.raises(ValueError):
        ACParams(T=-1.0)
    with pytest.raises(ValueError):
        ACParams(N=0)
    with pytest.raises(ValueError):
        ACParams(sigma=float("nan"))
    with pytest.raises(ValueError):
        ACParams(eta_t=0.0)
    with pytest.raises(ValueError):
        ACParams(eta_p=-0.1)
    with pytest.raises(ValueError):
        ACParams(rho=0.0)


def test_trajectory_endpoints_and_shape() -> None:
    p = ACParams()
    x = optimal_trajectory(p, lam=1.0)
    assert x.shape == (p.N + 1,)
    assert x[0] == p.X
    assert x[-1] == 0.0
    assert np.all(np.diff(x) <= 0.0)  # pure liquidation, never buys
    n = trade_list(p, lam=1.0)
    assert n.shape == (p.N,)
    assert n.sum() == pytest.approx(p.X)


def test_twap_limit() -> None:
    """lam -> 0 must collapse to the linear schedule, not divide by zero."""
    p = ACParams()
    x = optimal_trajectory(p, lam=0.0)
    expected = p.X * (p.T - np.linspace(0.0, p.T, p.N + 1)) / p.T
    np.testing.assert_allclose(x, expected)


def test_risk_aversion_frontloads() -> None:
    """Higher lambda sells faster early — AS/AC skew direction."""
    p = ACParams()
    x_lo = optimal_trajectory(p, lam=0.1)
    x_hi = optimal_trajectory(p, lam=1000.0)
    mid = p.N // 2
    assert x_hi[mid] < x_lo[mid]


def test_frontier_direction() -> None:
    """E up, Var down as lambda grows — the AC tradeoff."""
    p = ACParams()
    fr = efficient_frontier(p, lambdas=(0.1, 1.0, 10.0, 100.0))
    e = np.asarray(fr["expected_cost"].to_list())
    v = np.asarray(fr["var_cost"].to_list())
    assert np.all(np.diff(e) > 0)
    assert np.all(np.diff(v) < 0)


def test_frontier_point_mc_matches_theory() -> None:
    """MC mean cost agrees with the closed-form expectation."""
    p = ACParams(N=20)
    fp = frontier_point(p, lam=10.0, n_paths=512, seed=0)
    rel = abs(fp["mc_mean_cost"] - fp["expected_cost"]) / fp["expected_cost"]
    assert rel < 0.05
    var_rel = abs(fp["mc_var_cost"] - fp["var_cost"]) / fp["var_cost"]
    assert var_rel < 0.25


def test_transient_decay_monotone_and_bounded() -> None:
    """Finite rho stays feasible (sums to X, nonincreasing) and differs
    from the classic schedule only through impact memory."""
    p = ACParams(rho=4.0)
    x = optimal_trajectory(p, lam=1.0)
    assert x[0] == pytest.approx(p.X)
    assert x[-1] == 0.0
    assert np.all(np.diff(x) <= 1e-9)
    # transient memory smooths the profile vs the memoryless optimum
    x_classic = optimal_trajectory(ACParams(rho=math.inf), lam=1.0)
    assert not np.allclose(x, x_classic)


def test_transient_converges_to_classic_at_high_rho() -> None:
    """rho -> inf must recover the sinh schedule (kernel collapses)."""
    p_fast = ACParams(rho=200.0)
    p_classic = ACParams(rho=math.inf)
    lam = 10.0
    x_t = optimal_trajectory(p_fast, lam)
    x_c = optimal_trajectory(p_classic, lam)
    np.testing.assert_allclose(x_t, x_c, rtol=0.15, atol=p_classic.X * 0.02)


def test_bench_receipt_schema() -> None:
    out = ac_bench(n_mc_paths=64, seed=0)
    rec = out["receipt"]
    assert rec["schema"] == ALMGREN_CHRISS_SCHEMA
    assert rec["kind"] == "almgren_chriss"
    assert rec["data_label"] == "SYNTHETIC"
    m = rec["metrics"]
    assert m["frontier_monotone_expected"] is True
    assert m["frontier_monotone_var"] is True
    assert m["mc_mean_rel_err"] < 0.05
    assert set(m) >= {
        "frontier_monotone_expected",
        "frontier_monotone_var",
        "mc_mean_rel_err",
        "mc_var_rel_err",
        "mid_lam",
    }


def test_bench_fail_closed() -> None:
    with pytest.raises(ValueError):
        ac_bench(n_mc_paths=0)
    with pytest.raises(ValueError):
        frontier_point(ACParams(), lam=-1.0)


def test_determinism() -> None:
    a = ac_bench(n_mc_paths=64, seed=5)
    b = ac_bench(n_mc_paths=64, seed=5)
    assert a["frame"].equals(b["frame"])
    assert a["receipt"]["metrics"] == b["receipt"]["metrics"]
