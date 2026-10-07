import numpy as np
import pytest

from quant_fund.flowbars.bars import (
    bar_last_prices,
    bar_ohlc,
    bar_returns,
    dollar_bar_ids,
    synth_tape,
    tick_bar_ids,
    volume_bar_ids,
)

pytestmark = pytest.mark.synthetic


def test_synth_tape_is_deterministic() -> None:
    a = synth_tape(500, seed=7)
    b = synth_tape(500, seed=7)
    np.testing.assert_array_equal(a["price"], b["price"])
    np.testing.assert_array_equal(a["size"], b["size"])
    assert a["price"].shape == (500,)
    assert np.all(a["size"] >= 1.0)
    assert np.all(a["dollar"] == a["price"] * a["size"])


def test_dollar_bar_ids_cross_threshold() -> None:
    dollar = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ids = dollar_bar_ids(dollar, 2.0)
    # cumulative: 1,2,3,4,5,6 → bars 0,1,1,2,2,3
    np.testing.assert_array_equal(ids, [0, 1, 1, 2, 2, 3])


def test_dollar_bar_ids_monotone_and_dense() -> None:
    tape = synth_tape(5000, seed=1)
    ids = dollar_bar_ids(tape["dollar"], threshold=2000.0)
    assert np.all(np.diff(ids) >= 0)
    assert ids[0] == 0
    # dense: every integer bar id between 0 and max appears
    unique = np.unique(ids)
    np.testing.assert_array_equal(unique, np.arange(unique[-1] + 1))


def test_higher_threshold_yields_fewer_bars() -> None:
    tape = synth_tape(5000, seed=2)
    small = dollar_bar_ids(tape["dollar"], 500.0)
    large = dollar_bar_ids(tape["dollar"], 5000.0)
    assert small[-1] > large[-1]


def test_volume_bar_ids() -> None:
    sizes = np.array([3.0, 3.0, 3.0, 3.0])
    ids = volume_bar_ids(sizes, 6.0)
    # the crossing trade opens the new bar: cum 3,6,9,12 → bars 0,1,1,2
    np.testing.assert_array_equal(ids, [0, 1, 1, 2])


def test_tick_bar_ids() -> None:
    ids = tick_bar_ids(10, 3)
    np.testing.assert_array_equal(ids, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3])


def test_bar_ohlc_matches_manual() -> None:
    prices = np.array([1.0, 2.0, 1.5, 4.0, 3.0, 2.5])
    ids = np.array([0, 0, 0, 1, 1, 1])
    ohlc = bar_ohlc(prices, ids)
    np.testing.assert_allclose(ohlc["open"], [1.0, 4.0])
    np.testing.assert_allclose(ohlc["high"], [2.0, 4.0])
    np.testing.assert_allclose(ohlc["low"], [1.0, 2.5])
    np.testing.assert_allclose(ohlc["close"], [1.5, 2.5])


def test_bar_last_prices_and_returns() -> None:
    prices = np.array([10.0, 11.0, 12.0, 13.0])
    ids = np.array([0, 0, 1, 1])
    np.testing.assert_allclose(bar_last_prices(prices, ids), [11.0, 13.0])
    rets = bar_returns(prices, ids)
    np.testing.assert_allclose(rets, [13.0 / 11.0 - 1.0])


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        dollar_bar_ids(np.array([1.0, 2.0]), 0.0)
    with pytest.raises(ValueError):
        tick_bar_ids(5, 0)
    with pytest.raises(ValueError):
        synth_tape(1)
    with pytest.raises(ValueError):
        bar_ohlc(np.array([1.0]), np.array([0, 0]))
