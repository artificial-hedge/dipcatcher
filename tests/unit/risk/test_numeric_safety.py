"""Fail-closed numeric safety for risk, sizing, and portfolio math.

These checks refuse a usable size when an input is non-finite, a reduction
is empty in a way that used to zero-fill, or a solver returns a cancelled
book. Research diagnostics that already return an honest NaN stay on that
contract and are not retested here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from quant_fund.metrics.drawdown import ulcer_index
from quant_fund.metrics.perf_ratios import adjusted_sharpe_ratio, m_squared
from quant_fund.metrics.returns import downside_deviation
from quant_fund.portfolio import allocators
from quant_fund.portfolio.allocators import black_litterman
from quant_fund.portfolio.resampled import max_sharpe_weights, resampled_weights
from quant_fund.risk.gates import crash_leverage, dd_halt, vol_target_leverage
from quant_fund.risk.pyriskmgmt import component_var, portfolio_returns, scale_weights_to_es


def test_portfolio_returns_reject_nonfinite_inputs() -> None:
    book = np.full((4, 2), 0.01)
    weights = np.array([0.25, 0.75])
    poisoned = book.copy()
    poisoned[2, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        portfolio_returns(poisoned, weights)
    with pytest.raises(ValueError, match="finite"):
        portfolio_returns(book, np.array([0.5, np.nan]))
    empty = portfolio_returns(np.zeros((0, 2)), weights)
    assert empty.shape == (0,)


def test_component_var_rejects_nonfinite_long_book() -> None:
    book = np.full((10, 2), 0.01)
    book[4, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        component_var(book, np.array([0.4, 0.6]))
    # Short or mismatched books stay on the documented all-NaN contract.
    assert np.isnan(component_var(book[:5], np.array([0.4, 0.6]))).all()
    clean = np.full((10, 2), 0.01)
    with pytest.raises(ValueError, match="alpha"):
        component_var(clean, np.array([0.4, 0.6]), alpha=float("nan"))


def test_scale_weights_refuses_unmeasurable_es_and_keeps_nonpositive_es() -> None:
    short = np.full((3, 2), 0.01)
    weights = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="predicted ES"):
        scale_weights_to_es(short, weights, es_limit=0.01, method="ewma")
    # A book of pure gains has a negative loss-tail. That is not a missing
    # measurement, so the cap does not invent a scale.
    gains = np.full((30, 2), 0.01)
    scaled, blob = scale_weights_to_es(gains, weights, es_limit=0.001, method="historical")
    assert blob["capped"] is False
    assert blob["scale"] == 1.0
    np.testing.assert_allclose(scaled, weights)


def test_downside_deviation_rejects_nonfinite_and_keeps_short_nan() -> None:
    with pytest.raises(ValueError, match="returns must be finite"):
        downside_deviation(np.array([0.01, np.inf]))
    with pytest.raises(ValueError, match="returns must be finite"):
        downside_deviation(np.array([np.nan, -0.02]))
    assert math.isnan(downside_deviation(np.array([0.01])))
    assert downside_deviation(np.array([0.02, 0.03])) == 0.0


def test_drawdown_rejects_nonfinite_and_ruin() -> None:
    with pytest.raises(ValueError, match="finite"):
        ulcer_index(np.concatenate([np.full(20, 0.01), [np.nan]]))
    ruined = np.full(20, 0.01)
    ruined[0] = -1.0
    with pytest.raises(ValueError, match="ruin"):
        ulcer_index(ruined)


def test_perf_ratios_reject_nonfinite_scalars() -> None:
    path = np.linspace(-0.01, 0.02, 30)
    with pytest.raises(ValueError, match="rf must be finite"):
        adjusted_sharpe_ratio(path, rf=float("nan"))
    with pytest.raises(ValueError, match="periods_per_year must be finite and positive"):
        m_squared(path, path, periods_per_year=0.0)
    with pytest.raises(ValueError, match="periods_per_year must be finite"):
        m_squared(path, path, periods_per_year=float("inf"))


def test_dd_halt_nonfinite_stays_cash_and_negative_limit_is_a_magnitude() -> None:
    out = dd_halt(np.array([0.01, np.nan, 0.50]), limit=0.50)
    np.testing.assert_array_equal(out, [0.01, 0.0, 0.0])
    path = np.array([0.0, -0.05, 0.03])
    np.testing.assert_array_equal(dd_halt(path, limit=-0.05), dd_halt(path, limit=0.05))


def test_vol_target_leverage_nonfinite_window_is_flat() -> None:
    returns = np.array([0.01, -0.02, 0.03, np.nan, 0.01])
    lev = vol_target_leverage(returns, target=0.10, lookback=3, periods_per_year=1.0, cap=5.0)
    prior = min(1.0, 0.10 / 0.20)
    assert lev[0] == prior
    assert lev[1] == prior
    assert lev[3] == 0.0
    assert lev[4] == 0.0
    with pytest.raises(ValueError, match="vol cap must be finite and positive"):
        vol_target_leverage(returns[:3], cap=float("inf"))


def test_crash_leverage_rejects_nonfinite_threshold() -> None:
    with pytest.raises(ValueError, match="crash_return must be finite"):
        crash_leverage(np.zeros(5), crash_return=float("nan"))


def test_black_litterman_rejects_nonfinite_delta() -> None:
    with pytest.raises(ValueError, match="delta must be finite"):
        black_litterman(np.eye(3), delta=float("nan"))
    with pytest.raises(ValueError, match="delta must be finite"):
        black_litterman(np.eye(3), delta=float("inf"))


def test_cvar_long_short_zero_net_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_linprog(*_args: object, **_kwargs: object) -> OptimizeResult:
        # n=2, T=5 → [weights, zeta, five shortfall auxiliaries].
        solved = np.zeros(8)
        solved[0] = 1.0
        solved[1] = -1.0
        return OptimizeResult(x=solved, success=True, message="cancel")

    monkeypatch.setattr(allocators.opt, "linprog", fake_linprog)
    returns = np.full((5, 2), 0.01)
    with pytest.raises(ValueError, match="zero-net"):
        allocators.cvar_minimization(returns, long_only=False)


def test_max_sharpe_does_not_treat_zero_variance_as_better() -> None:
    mu = np.array([-1.0, 0.0])
    cov = np.diag([1.0, 0.0])
    weights = max_sharpe_weights(mu, cov)
    assert abs(float(weights.sum()) - 1.0) < 1e-8
    assert float(weights @ cov @ weights) > 1e-8


def test_resampled_zero_sum_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from quant_fund.portfolio import resampled as resampled_mod

    monkeypatch.setattr(
        resampled_mod, "max_sharpe_weights", lambda *_a, **_k: np.array([1.0, -1.0])
    )
    with pytest.raises(ValueError, match="summed to zero"):
        resampled_weights(np.array([0.05, 0.04]), np.eye(2), n_obs=10, n_sims=2)
