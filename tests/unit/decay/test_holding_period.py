import numpy as np
import pytest

from quant_fund.decay.holding_period import (
    lag_breakeven_table,
    net_ic_curve,
    optimal_holding_lag,
)

pytestmark = pytest.mark.synthetic


def _profile():
    ic = np.array([0.12, 0.10, 0.075, 0.05, 0.03])
    turnover = np.array([0.90, 0.45, 0.30, 0.22, 0.18])
    return ic, turnover


def test_net_ic_curve_linear_in_cost() -> None:
    ic, to = _profile()
    np.testing.assert_allclose(net_ic_curve(ic, to, 0.1), ic - 0.1 * to)
    np.testing.assert_allclose(net_ic_curve(ic, to, 0.0), ic)


def test_optimal_lag_shifts_with_cost() -> None:
    ic, to = _profile()
    cheap = optimal_holding_lag(ic, to, 0.02)
    pricey = optimal_holding_lag(ic, to, 0.30)
    assert cheap["lag"] == 1.0
    assert pricey["lag"] > cheap["lag"]
    curve = cheap["curve"]
    assert isinstance(curve, np.ndarray)
    assert curve.shape == ic.shape


def test_breakeven_table() -> None:
    ic, to = _profile()
    be = lag_breakeven_table(ic, to)
    np.testing.assert_allclose(be, ic / to)
    assert np.all(np.isfinite(be))
    neg = lag_breakeven_table(np.array([-0.1, 0.1]), np.array([0.2, 0.2]))
    assert neg[0] == float("inf")


def test_validation() -> None:
    with pytest.raises(ValueError):
        net_ic_curve(np.zeros(3), np.zeros(4), 0.1)
    with pytest.raises(ValueError):
        optimal_holding_lag(np.full(3, np.nan), np.zeros(3), 0.1)
