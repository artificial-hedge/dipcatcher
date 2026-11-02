"""features/bars.py — alternative bar constructors and their argument guards.

Ticks are SYNTHETIC; the assertions pin bar-boundary semantics, not alpha.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.features import bars

KEYS = {"start", "open", "high", "low", "close", "volume", "dollar"}


def _ticks(n: int = 120, seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    prices = 50.0 + np.cumsum(rng.normal(0.0, 0.05, n))
    sizes = rng.uniform(1.0, 10.0, n)
    return prices, sizes


class TestTickBars:
    def test_bars_partition_the_ticks(self) -> None:
        p, s = _ticks()
        out = bars.tick_bars(p, s, ticks_per_bar=10)
        assert set(out) == KEYS
        assert out["start"].size == 12
        assert out["volume"].sum() == pytest.approx(s[:120].sum())
        assert np.all(out["high"] >= out["low"])

    @pytest.mark.parametrize("per_bar", [0, -1, True, 2.5])
    def test_invalid_bar_size_rejected(self, per_bar: object) -> None:
        p, s = _ticks()
        with pytest.raises(ValueError, match="positive integer"):
            bars.tick_bars(p, s, ticks_per_bar=per_bar)  # type: ignore[arg-type]

    def test_too_few_ticks_rejected(self) -> None:
        with pytest.raises(ValueError, match="not enough ticks"):
            bars.tick_bars(np.array([1.0, 2.0]), np.array([1.0, 1.0]), ticks_per_bar=10)


class TestTickGuards:
    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            bars.tick_bars(np.array([1.0, 2.0]), np.array([1.0]), ticks_per_bar=1)

    def test_non_finite_ticks_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            bars.tick_bars(np.array([1.0, np.nan]), np.array([1.0, 1.0]), ticks_per_bar=1)

    def test_non_positive_price_rejected(self) -> None:
        with pytest.raises(ValueError, match="prices must be positive"):
            bars.tick_bars(np.array([1.0, 0.0]), np.array([1.0, 1.0]), ticks_per_bar=1)

    def test_negative_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="sizes non-negative"):
            bars.tick_bars(np.array([1.0, 2.0]), np.array([1.0, -1.0]), ticks_per_bar=1)


class TestVolumeAndDollarBars:
    def test_volume_bars_split_on_cumulative_volume(self) -> None:
        p = np.full(20, 10.0)
        s = np.full(20, 1.0)
        out = bars.volume_bars(p, s, volume_per_bar=5.0)
        assert set(out) == KEYS
        assert np.all(out["volume"] == pytest.approx(5.0))

    def test_dollar_bars_split_on_cumulative_notional(self) -> None:
        p = np.full(20, 10.0)
        s = np.full(20, 1.0)
        out = bars.dollar_bars(p, s, dollar_per_bar=20.0)
        assert np.all(out["dollar"] == pytest.approx(20.0))

    @pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf")])
    def test_volume_threshold_guard(self, threshold: float) -> None:
        p, s = _ticks()
        with pytest.raises(ValueError, match="volume_per_bar must be positive and finite"):
            bars.volume_bars(p, s, threshold)

    @pytest.mark.parametrize("threshold", [0.0, -5.0, float("nan")])
    def test_dollar_threshold_guard(self, threshold: float) -> None:
        p, s = _ticks()
        with pytest.raises(ValueError, match="dollar_per_bar must be positive and finite"):
            bars.dollar_bars(p, s, threshold)

    def test_no_complete_bar_rejected(self) -> None:
        p = np.full(4, 10.0)
        s = np.full(4, 1.0)
        with pytest.raises(ValueError, match="no complete volume bars"):
            bars.volume_bars(p, s, volume_per_bar=100.0)
        with pytest.raises(ValueError, match="no complete dollar bars"):
            bars.dollar_bars(p, s, dollar_per_bar=10_000.0)


class TestImbalanceAndRunBars:
    def test_tick_imbalance_bars_cover_the_series(self) -> None:
        p, s = _ticks()
        out = bars.tick_imbalance_bars(p, s, expected_ticks=10, alpha_ewma=0.2)
        assert set(out) == KEYS
        assert out["start"].size >= 1
        assert out["start"][0] == 0.0

    def test_tick_run_bars_cover_the_series(self) -> None:
        p, s = _ticks()
        out = bars.tick_run_bars(p, s, expected_ticks=10, alpha_ewma=0.2)
        assert out["start"][0] == 0.0
        assert np.all(np.diff(out["start"]) > 0)

    def test_dollar_imbalance_bars_cover_the_series(self) -> None:
        p, s = _ticks()
        out = bars.dollar_imbalance_bars(p, s, expected_ticks=2, alpha_ewma=0.9)
        assert out["start"][0] == 0.0
        assert out["start"].size > 1
        assert np.all(np.diff(out["start"]) > 0)

    @pytest.mark.parametrize("expected", [0, 1, -3])
    def test_expected_ticks_guard(self, expected: int) -> None:
        p, s = _ticks()
        with pytest.raises(ValueError, match=r"expected_ticks >= 2"):
            bars.tick_imbalance_bars(p, s, expected_ticks=expected)

    @pytest.mark.parametrize("alpha", [0.0, 1.5, -0.1, float("nan")])
    def test_alpha_guard(self, alpha: float) -> None:
        p, s = _ticks()
        with pytest.raises(ValueError, match=r"alpha_ewma in \(0,1\]"):
            bars.tick_run_bars(p, s, alpha_ewma=alpha)

    def test_no_complete_imbalance_bar_rejected(self) -> None:
        p = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        s = np.ones(5)
        with pytest.raises(ValueError, match="no complete run bars"):
            bars.tick_run_bars(p, s, expected_ticks=1000, alpha_ewma=0.5)
