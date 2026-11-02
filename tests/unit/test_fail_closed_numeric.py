"""NaN, infinity, and negative sizes, prices, and leverage fail closed."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from quant_fund.backtest.carry_engine import run_carry_backtest
from quant_fund.backtest.engine import run_backtest
from quant_fund.backtest.fast_replay import run_backtest_fast
from quant_fund.backtest.inputs import finite_non_negative, finite_positive
from quant_fund.backtest.perp_engine import run_perp_backtest
from quant_fund.backtest.sleeves import _cap_and_emit
from quant_fund.config.models import AppConfig
from quant_fund.flowbars.bars import dollar_bar_ids, tick_bar_ids, volume_bar_ids
from quant_fund.flowbars.imbalance import imbalance_bar_ids, tick_rule_signs
from quant_fund.flowbars.information import information_bar_ids
from quant_fund.flowbars.runs import run_bar_ids

pytestmark = pytest.mark.synthetic

_BAD = (float("nan"), float("inf"), float("-inf"), -1.0, True)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return AppConfig()


def test_zero_capital_stays_allowed() -> None:
    assert finite_non_negative(0.0, "initial_nav") == 0.0
    assert finite_non_negative(np.float64(0.0), "initial_nav") == 0.0


@pytest.mark.parametrize("nav", _BAD)
@pytest.mark.parametrize("runner", (run_backtest, run_backtest_fast))
def test_spot_engines_reject_bad_initial_nav(runner, config: AppConfig, nav: object) -> None:
    bars = pl.DataFrame()
    weights = pl.DataFrame()
    with pytest.raises(ValueError, match="initial_nav must be finite and non-negative"):
        runner(bars, weights, config, initial_nav=nav)


@pytest.mark.parametrize("nav", _BAD)
def test_perp_rejects_bad_initial_nav(config: AppConfig, nav: object) -> None:
    empty = pl.DataFrame()
    with pytest.raises(ValueError, match="initial_nav must be finite and non-negative"):
        run_perp_backtest(empty, None, empty, config, initial_nav=nav)


@pytest.mark.parametrize("nav", _BAD)
def test_carry_rejects_bad_initial_nav(config: AppConfig, nav: object) -> None:
    empty = pl.DataFrame()
    with pytest.raises(ValueError, match="initial_nav must be finite and non-negative"):
        run_carry_backtest(empty, empty, None, empty, config, initial_nav=nav)


@pytest.mark.parametrize("buffer", _BAD)
def test_carry_rejects_bad_cash_buffer(config: AppConfig, buffer: object) -> None:
    empty = pl.DataFrame()
    with pytest.raises(ValueError, match="cash_buffer_frac must be finite and non-negative"):
        run_carry_backtest(empty, empty, None, empty, config, cash_buffer_frac=buffer)


@pytest.mark.parametrize("scale", (float("nan"), float("inf"), float("-inf"), 0.0, -2.0, True))
def test_leverage_scale_rejects_bad_bounds(scale: object) -> None:
    frame = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1)],
            "security_id": ["AAA"],
            "_raw": [0.2],
        }
    )
    with pytest.raises(ValueError, match="must be finite and positive"):
        _cap_and_emit(frame, "_raw", max_name=0.5, gross_scale=scale)
    with pytest.raises(ValueError, match="must be finite and positive"):
        _cap_and_emit(frame, "_raw", max_name=scale, gross_scale=1.0)


def test_leverage_scale_keeps_a_finite_weight() -> None:
    frame = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1)],
            "security_id": ["AAA"],
            "_raw": [0.2],
        }
    )
    out = _cap_and_emit(frame, "_raw", max_name=0.5, gross_scale=1.0)
    assert out["target_weight"].to_list() == [0.2]
    assert finite_positive(1.0, "gross_scale") == 1.0


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf"), -1.0))
def test_volume_and_dollar_bars_reject_bad_sizes(bad: float) -> None:
    good = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="sizes must be"):
        volume_bar_ids(np.array([1.0, bad, 3.0]), 2.0)
    with pytest.raises(ValueError, match="dollar must be"):
        dollar_bar_ids(np.array([1.0, bad, 3.0]), 2.0)
    with pytest.raises(ValueError, match="threshold must be finite and positive"):
        volume_bar_ids(good, bad)


def test_volume_bars_still_cut_on_a_finite_threshold() -> None:
    ids = volume_bar_ids(np.array([1.0, 2.0, 3.0]), 3.0)
    assert ids.tolist() == [0, 1, 2]


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), 0.0, -2.0))
def test_prices_and_thresholds_reject_non_positive(bad: float) -> None:
    with pytest.raises(ValueError, match="prices must be"):
        information_bar_ids(np.array([100.0, bad]), 0.01)
    with pytest.raises(ValueError, match="prices must be"):
        tick_rule_signs(np.array([10.0, bad]))
    with pytest.raises(ValueError, match="threshold must be finite and positive"):
        information_bar_ids(np.array([100.0, 101.0, 102.0]), bad)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf"), -2.0))
def test_flow_sizes_reject_non_finite_and_negative(bad: float) -> None:
    with pytest.raises(ValueError, match="values must be"):
        imbalance_bar_ids(np.array([1, -1]), np.array([bad, 1.0]), 1.0)
    with pytest.raises(ValueError, match="values must be"):
        run_bar_ids(np.array([1, -1]), np.array([1.0, bad]), 1.0)


def test_tick_stride_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="every must be a positive integer"):
        tick_bar_ids(4, 0)
    with pytest.raises(ValueError, match="every must be a positive integer"):
        tick_bar_ids(4, -1)
    with pytest.raises(ValueError, match="n_trades must be a non-negative integer"):
        tick_bar_ids(-1, 2)
    assert tick_bar_ids(4, 2).tolist() == [0, 0, 1, 1]
