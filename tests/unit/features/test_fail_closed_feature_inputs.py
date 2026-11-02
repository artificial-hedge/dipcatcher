"""Fail-closed feature inputs: bad prices, thresholds, and winsor levels.

SYNTHETIC rows only. These checks guard construction, not a market result.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.features.bars import dollar_bars, tick_imbalance_bars, tick_run_bars, volume_bars
from quant_fund.features.cross_sectional import apply_cross_sectional
from quant_fund.features.engine import compute_base_features


def _bars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2024, 1, 2)],
            "close_total_return": [100.0],
            "close": [100.0],
            "volume": [1_000.0],
            "open_split_adjusted": [99.0],
            "close_split_adjusted": [100.0],
            "high_split_adjusted": [101.0],
            "low_split_adjusted": [98.0],
        }
    )


def test_usable_ohlcv_still_builds() -> None:
    out = compute_base_features(_bars(), AppConfig())
    assert out.height == 1
    assert out["ret_1"].null_count() == 1


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("close", float("nan")),
        ("close", float("inf")),
        ("close", float("-inf")),
        ("close", 0.0),
        ("close", -1.0),
        ("close_total_return", float("nan")),
        ("open_split_adjusted", 0.0),
        ("low_split_adjusted", float("inf")),
        ("volume", float("nan")),
        ("volume", float("inf")),
        ("volume", -1.0),
    ],
)
def test_compute_base_features_rejects_unusable_ohlcv(column: str, value: float) -> None:
    bars = _bars().with_columns(pl.lit(value).alias(column))
    with pytest.raises(ValueError, match="finite"):
        compute_base_features(bars, AppConfig())


def test_compute_base_features_rejects_null_price() -> None:
    bars = _bars().with_columns(pl.lit(None, dtype=pl.Float64).alias("close"))
    with pytest.raises(ValueError, match="finite"):
        compute_base_features(bars, AppConfig())


def test_compute_base_features_rejects_non_numeric_volume() -> None:
    bars = _bars().with_columns(pl.lit("1000").alias("volume"))
    with pytest.raises(ValueError, match="volume must be numeric"):
        compute_base_features(bars, AppConfig())


def test_zero_volume_is_a_real_bar() -> None:
    bars = _bars().with_columns(pl.lit(0.0).alias("volume"))
    out = compute_base_features(bars, AppConfig())
    assert out.height == 1
    assert out["dollar_volume"][0] == 0.0


def _panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 2, tzinfo=UTC)],
            "x": [1.0],
        }
    )


def test_winsor_zero_still_runs() -> None:
    out = apply_cross_sectional(_panel(), ["x"], 0.0)
    assert out.height == 1
    assert "winsor_x" in out.columns


@pytest.mark.parametrize(
    "level", [float("nan"), float("inf"), float("-inf"), 0.5, -0.01, True, "0.1"]
)
def test_winsor_rejects_unusable_level(level: object) -> None:
    with pytest.raises(ValueError, match="winsor_p"):
        apply_cross_sectional(_panel(), ["x"], level)  # type: ignore[arg-type]


def _ticks() -> tuple[np.ndarray, np.ndarray]:
    prices = np.array([100.0, 101.0, 99.0, 102.0])
    sizes = np.array([1.0, 1.0, 1.0, 1.0])
    return prices, sizes


@pytest.mark.parametrize("threshold", [True, float("nan"), float("inf"), 0.0, -1.0, "10"])
def test_volume_and_dollar_thresholds_fail_closed(threshold: object) -> None:
    prices, sizes = _ticks()
    with pytest.raises(ValueError, match="volume_per_bar"):
        volume_bars(prices, sizes, threshold)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dollar_per_bar"):
        dollar_bars(prices, sizes, threshold)  # type: ignore[arg-type]


def test_positive_bar_threshold_still_runs() -> None:
    prices, sizes = _ticks()
    assert volume_bars(prices, sizes, 2.0)["close"].size >= 1
    assert dollar_bars(prices, sizes, 150.0)["close"].size >= 1


@pytest.mark.parametrize("alpha", [True, float("nan"), float("inf"), 0.0, 1.5])
def test_imbalance_alpha_fails_closed(alpha: object) -> None:
    prices, sizes = _ticks()
    with pytest.raises(ValueError, match="alpha_ewma"):
        tick_imbalance_bars(prices, sizes, expected_ticks=2, alpha_ewma=alpha)  # type: ignore[arg-type]


def test_expected_ticks_rejects_non_integers() -> None:
    prices, sizes = _ticks()
    with pytest.raises(ValueError, match="expected_ticks"):
        tick_run_bars(prices, sizes, expected_ticks=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected_ticks"):
        tick_run_bars(prices, sizes, expected_ticks=True)  # type: ignore[arg-type]
