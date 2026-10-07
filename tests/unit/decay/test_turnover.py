import numpy as np
import pytest

from quant_fund.decay.turnover import (
    breakeven_cost,
    cost_adjusted_ic,
    signal_turnover,
    signal_weights,
)

pytestmark = pytest.mark.synthetic


def test_signal_weights_gross_and_direction() -> None:
    pred = np.array([[1.0, 2.0, 3.0, 4.0]])
    w = signal_weights(pred, gross=1.0)
    assert w.sum() == pytest.approx(0.0)
    assert np.sum(np.abs(w)) == pytest.approx(1.0)
    assert w[0, 3] > 0 and w[0, 0] < 0


def test_signal_weights_constant_row_is_zero() -> None:
    pred = np.array([[2.0, 2.0, 2.0]])
    w = signal_weights(pred)
    np.testing.assert_array_equal(w, np.zeros((1, 3)))


def test_turnover_zero_for_constant_weights_after_entry() -> None:
    w = np.tile(np.array([0.25, 0.25, -0.25, -0.25]), (5, 1))
    to = signal_turnover(w)
    # period 0 is the entry (½Σ|w| = 0.5), later periods do not trade
    np.testing.assert_allclose(to, [0.5, 0.0, 0.0, 0.0, 0.0])


def test_turnover_first_period_is_entry() -> None:
    w = np.array([[0.5, -0.5]])
    assert signal_turnover(w)[0] == pytest.approx(0.5)


def test_turnover_positive_after_flip() -> None:
    w = np.array([[0.5, -0.5], [-0.5, 0.5]])
    assert signal_turnover(w)[1] == pytest.approx(1.0)


def test_cost_adjusted_ic_linear_in_cost() -> None:
    ic = np.full(10, 0.1)
    to = np.full(10, 0.2)
    np.testing.assert_allclose(cost_adjusted_ic(ic, to, 0.25), 0.1 - 0.25 * 0.2)
    np.testing.assert_allclose(cost_adjusted_ic(ic, to, 0.5), 0.0)


def test_breakeven_cost() -> None:
    ic = np.full(10, 0.1)
    to = np.full(10, 0.2)
    assert breakeven_cost(ic, to) == pytest.approx(0.5)


def test_breakeven_zero_turnover_is_inf() -> None:
    assert breakeven_cost(np.full(5, 0.1), np.zeros(5)) == float("inf")


def test_shape_validation() -> None:
    with pytest.raises(ValueError):
        signal_weights(np.zeros((3,)))
    with pytest.raises(ValueError):
        cost_adjusted_ic(np.zeros(3), np.zeros(4), 0.1)
