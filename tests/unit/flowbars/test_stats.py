import numpy as np
import pytest

from quant_fund.flowbars.bars import (
    bar_returns,
    dollar_bar_ids,
    synth_tape,
    tick_bar_ids,
)
from quant_fund.flowbars.stats import (
    bar_return_stats,
    bars_per_period,
    compare_bar_returns,
    durbin_watson,
    lag1_serial_corr,
    variance_ratio,
)

pytestmark = pytest.mark.synthetic


def test_variance_ratio_iid_near_one() -> None:
    rng = np.random.default_rng(10)
    r = rng.standard_normal(4000)
    assert variance_ratio(r, horizon=2) == pytest.approx(1.0, abs=0.15)


def test_variance_ratio_persistent_above_one() -> None:
    rng = np.random.default_rng(11)
    x = np.empty(4000)
    x[0] = 0.0
    e = rng.standard_normal(4000)
    for i in range(1, 4000):
        x[i] = 0.9 * x[i - 1] + e[i]
    assert variance_ratio(x, horizon=2) > 1.2


def test_lag1_serial_corr_matches_ar1_rho() -> None:
    rng = np.random.default_rng(12)
    x = np.empty(4000)
    x[0] = 0.0
    e = rng.standard_normal(4000)
    rho = 0.6
    for i in range(1, 4000):
        x[i] = rho * x[i - 1] + e[i]
    assert lag1_serial_corr(x) == pytest.approx(rho, abs=0.08)


def test_durbin_watson_iid_near_two() -> None:
    rng = np.random.default_rng(13)
    assert durbin_watson(rng.standard_normal(3000)) == pytest.approx(2.0, abs=0.15)


def test_bar_return_stats_keys() -> None:
    rng = np.random.default_rng(14)
    stats = bar_return_stats(rng.standard_normal(500))
    for key in (
        "n",
        "mean",
        "std",
        "skew",
        "excess_kurtosis",
        "serial_corr_lag1",
        "variance_ratio_2",
        "durbin_watson",
    ):
        assert key in stats
    assert stats["n"] == 500


def test_bars_per_period() -> None:
    # bars [0,0],[1,1],[2,2],[3,3],[4,4] close at idx 1,3,5,7,9
    ids = tick_bar_ids(10, 2)
    counts = bars_per_period(ids, period_length=4)
    # period 0: closes at idx 1,3 → 2 bars; period 1: idx 5,7 → 2; period 2: 1
    np.testing.assert_array_equal(counts, [2, 2, 1])


def test_compare_bar_returns_reports_reductions() -> None:
    tape = synth_tape(4000, seed=15)
    r_time = bar_returns(tape["price"], tick_bar_ids(len(tape["price"]), 50))
    r_dollar = bar_returns(tape["price"], dollar_bar_ids(tape["dollar"], 2000.0))
    out = compare_bar_returns(r_time, r_dollar)
    assert set(out) == {"time", "alternative", "reduction_serial_corr", "reduction_excess_kurtosis"}
    assert out["time"]["n"] == len(r_time)
    assert out["alternative"]["n"] == len(r_dollar)


def test_compare_needs_three_points() -> None:
    with pytest.raises(ValueError):
        bar_return_stats(np.array([0.01, 0.02]))
