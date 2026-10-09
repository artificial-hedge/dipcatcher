import numpy as np
import pytest

from quant_fund.flowbars.bar_features import (
    bar_amihud,
    bar_body,
    bar_feature_frame,
    bar_log_range,
    bar_path_entropy,
    bar_signed_volume,
    bar_wick_asymmetry,
    close_location_value,
)
from quant_fund.flowbars.bars import bar_ohlc, dollar_bar_ids, synth_tape

pytestmark = pytest.mark.synthetic


def _toy_ohlc() -> dict[str, np.ndarray]:
    return {
        "open": np.array([10.0, 10.0, 10.0]),
        "high": np.array([12.0, 11.0, 10.5]),
        "low": np.array([9.0, 9.5, 9.8]),
        "close": np.array([11.0, 9.6, 10.4]),
    }


def test_bar_log_range() -> None:
    out = bar_log_range(_toy_ohlc())
    np.testing.assert_allclose(out, np.log([12.0 / 9.0, 11.0 / 9.5, 10.5 / 9.8]))


def test_bar_body() -> None:
    out = bar_body(_toy_ohlc())
    np.testing.assert_allclose(out, np.log([11.0 / 10.0, 9.6 / 10.0, 10.4 / 10.0]))


def test_bar_wick_asymmetry_bounds() -> None:
    out = bar_wick_asymmetry(_toy_ohlc())
    assert np.all(out >= -1.0)
    assert np.all(out <= 1.0)
    # bar 0: up = 12−11 = 1, lo = 10−9 = 1 → asym 0
    assert out[0] == pytest.approx(0.0)


def test_close_location_value() -> None:
    out = close_location_value(_toy_ohlc())
    # bar 0: (22−12−9)/(12−9) = 1/3
    assert out[0] == pytest.approx(1.0 / 3.0)


def test_bar_amihud_first_nan_and_positive() -> None:
    ohlc = _toy_ohlc()
    dv = np.array([1e6, 1e6, 1e6])
    out = bar_amihud(ohlc, dv)
    assert np.isnan(out[0])
    assert np.all(out[1:] > 0)


def test_bar_signed_volume_net_direction() -> None:
    prices = np.array([10.0, 11.0, 12.0, 11.5])
    sizes = np.array([1.0, 2.0, 3.0, 4.0])
    ids = np.array([0, 0, 1, 1])
    out = bar_signed_volume(prices, sizes, ids)
    # bar 0: signs +1,+1 → 3; bar 1: +1,−1 → −1
    np.testing.assert_allclose(out, [3.0, -1.0])


def test_bar_path_entropy_degenerate_bars_nan() -> None:
    prices = np.array([10.0, 10.0, 10.0])
    ids = np.array([0, 0, 0])
    out = bar_path_entropy(prices, ids, bins=4)
    assert np.isnan(out[0])  # zero returns → not enough variation


def test_bar_feature_frame_alignment() -> None:
    tape = synth_tape(3000, seed=20)
    ids = dollar_bar_ids(tape["dollar"], 1500.0)
    ohlc = bar_ohlc(tape["price"], ids)
    n_bars = int(ids[-1]) + 1
    dv = np.zeros(n_bars)
    np.add.at(dv, ids, tape["dollar"])
    frame = bar_feature_frame(ohlc, dv, tape["price"], tape["size"], ids)
    for key, arr in frame.items():
        assert arr.shape == (n_bars,), key
        assert arr.dtype == np.float64


def test_validation() -> None:
    with pytest.raises(ValueError):
        bar_log_range(
            {
                "open": np.array([1.0]),
                "high": np.array([1.0]),
                "low": np.array([2.0]),
                "close": np.array([1.0]),
            }
        )
    with pytest.raises(ValueError):
        bar_amihud(_toy_ohlc(), np.array([1.0]))
