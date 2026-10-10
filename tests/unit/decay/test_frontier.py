import numpy as np
import pytest

from quant_fund.decay.frontier import (
    efficiency_frontier,
    ic_turnover_points,
    trade_off_slope,
)

pytestmark = pytest.mark.synthetic


def test_frontier_keeps_non_dominated() -> None:
    turnover = np.array([0.9, 0.4, 0.6, 0.2])
    ic = np.array([0.10, 0.06, 0.11, 0.04])
    f = efficiency_frontier(turnover, ic)
    # order by turnover: (0.2, .04), (0.4, .06), (0.6, .11), (0.9, .10)
    # the last is dominated by (0.6, .11) → dropped
    np.testing.assert_array_equal(f, [3, 1, 2])


def test_frontier_all_dominated_except_best() -> None:
    turnover = np.array([0.5, 0.3, 0.1])
    ic = np.array([0.01, 0.05, 0.12])
    f = efficiency_frontier(turnover, ic)
    np.testing.assert_array_equal(f, [2])


def test_trade_off_slope_frontier_differences() -> None:
    turnover = np.array([0.2, 0.4, 0.9])
    ic = np.array([0.04, 0.06, 0.10])
    s = trade_off_slope(turnover, ic)
    assert s.shape == (2,)
    np.testing.assert_allclose(s, [0.02 / 0.2, 0.04 / 0.5])


def test_trade_off_slope_single_point() -> None:
    s = trade_off_slope(np.array([0.5]), np.array([0.1]))
    assert s.shape == (0,)


def test_ic_turnover_points_shape() -> None:
    rng = np.random.default_rng(90)
    t_total, n = 200, 40
    signals = [rng.standard_normal((t_total, n)), rng.standard_normal((t_total, n))]
    actual = signals[0] + rng.standard_normal((t_total, n))
    out = ic_turnover_points(signals, actual)
    for key in ("turnover", "ic", "ir"):
        assert out[key].shape == (2,)
    assert out["ic"][0] > out["ic"][1]


def test_validation() -> None:
    with pytest.raises(ValueError):
        efficiency_frontier(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError):
        ic_turnover_points([], np.zeros((5, 3)))
