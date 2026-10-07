import numpy as np
import pytest

from quant_fund.flowbars.imbalance import (
    adaptive_imbalance_bar_ids,
    imbalance_bar_ids,
    signed_flow_imbalance,
    tick_imbalance_bar_ids,
    tick_rule_signs,
)

pytestmark = pytest.mark.synthetic


def test_tick_rule_signs_basic() -> None:
    prices = np.array([10.0, 11.0, 11.0, 10.5, 10.5, 12.0])
    signs = tick_rule_signs(prices)
    np.testing.assert_array_equal(signs, [1, 1, 1, -1, -1, 1])


def test_imbalance_bar_ids_closes_at_threshold() -> None:
    signs = np.array([1, 1, 1, -1, -1, -1, 1])
    values = np.ones(7)
    ids = imbalance_bar_ids(signs, values, 3.0)
    # bars: [0,1,2] → close at idx 2; [-1,-1,-1] → close at idx 5; last open
    np.testing.assert_array_equal(ids, [0, 0, 0, 1, 1, 1, 2])


def test_imbalance_cancelling_flow_stays_open() -> None:
    signs = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    ids = tick_imbalance_bar_ids(signs, 2.0)
    # run: 1,0,1,0,1,0,1,0 → |run| never reaches 2
    np.testing.assert_array_equal(ids, np.zeros(8))


def test_tick_imbalance_cancelling_flow_closes() -> None:
    signs = np.array([1, 1, -1, -1, -1, -1])
    ids = tick_imbalance_bar_ids(signs, 2.0)
    # run: +1,+2 → close at idx1; −1,−2 → close at idx3; −1,−2 → close idx5
    np.testing.assert_array_equal(ids, [0, 0, 1, 1, 2, 2])


def test_adaptive_ids_monotone_and_nonempty() -> None:
    rng = np.random.default_rng(3)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 4000)))
    signs = tick_rule_signs(prices)
    sizes = rng.integers(1, 20, 4000).astype(np.float64)
    ids = adaptive_imbalance_bar_ids(signs, sizes, warm_up=10)
    assert ids[0] == 0
    assert np.all(np.diff(ids) >= 0)
    assert ids[-1] > 0


def test_signed_flow_imbalance_cumsum() -> None:
    signs = np.array([1, -1, 1], dtype=np.int64)
    sizes = np.array([2.0, 5.0, 3.0])
    out = signed_flow_imbalance(signs, sizes)
    np.testing.assert_allclose(out, [2.0, -3.0, 0.0])


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        imbalance_bar_ids(np.array([1, -1]), np.array([1.0]), 2.0)
    with pytest.raises(ValueError):
        tick_imbalance_bar_ids(np.array([1, -1]), -1.0)
