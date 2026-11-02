"""features/indicators.py — trend/momentum/volume blocks that lacked direct tests.

OHLCV fixtures are SYNTHETIC. Assertions pin shapes, warmup, and algebraic
relations (band ordering, VWAP identity), not market values.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.features import indicators

N = 160


def _bars(n: int = N, seed: int = 11) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.8, n))
    high = close + rng.uniform(0.1, 1.5, n)
    low = close - rng.uniform(0.1, 1.5, n)
    volume = rng.uniform(1_000.0, 5_000.0, n)
    return high, low, close, volume


class TestDmi:
    def test_all_four_lines_warm_up_then_fill(self) -> None:
        h, lo, c, _ = _bars()
        out = indicators.dmi(h, lo, c, window=14)
        assert set(out) == {"plus_di", "minus_di", "dx", "adx"}
        for series in out.values():
            assert series.shape == c.shape
            assert np.isfinite(series[60:]).all()
        assert np.isnan(out["adx"][:27]).all()
        assert np.nanmax(out["dx"]) <= 100.0


class TestAroon:
    def test_rising_highs_pin_aroon_up_at_100(self) -> None:
        h = np.arange(60.0)
        lo = h - 1.0
        out = indicators.aroon(h, lo, window=10)
        assert out["up"][-1] == pytest.approx(100.0)
        assert out["down"][-1] == pytest.approx(0.0)
        assert out["oscillator"][-1] == pytest.approx(100.0)

    def test_window_one_raises_division_by_zero(self) -> None:
        h = np.arange(60.0)
        with pytest.raises(ZeroDivisionError):
            indicators.aroon(h, h - 1.0, window=1)

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="match length"):
            indicators.aroon(np.arange(30.0), np.arange(29.0))


class TestVortex:
    def test_components_are_non_negative(self) -> None:
        h, lo, c, _ = _bars()
        out = indicators.vortex(h, lo, c, window=14)
        assert set(out) == {"plus", "minus"}
        for series in out.values():
            assert np.isfinite(series[20:]).all()
            assert np.all(series[20:] >= 0.0)


class TestTrix:
    def test_flat_series_stays_near_zero_or_nan(self) -> None:
        out = indicators.trix(np.full(80, 5.0), window=10)
        assert np.isnan(out[:20]).all()
        assert np.allclose(out[np.isfinite(out)], 0.0)

    def test_shape_and_warmup(self) -> None:
        out = indicators.trix(np.linspace(10.0, 20.0, 120), window=15)
        assert out.shape == (120,)
        assert np.isnan(out[:28]).all()


class TestCci:
    def test_monotone_series_saturates_positive(self) -> None:
        close = np.linspace(10.0, 30.0, 80)
        out = indicators.cci(close + 0.5, close - 0.5, close, window=20)
        assert out[-1] > 0.0
        assert np.isnan(out[:19]).all()

    def test_flat_bars_give_zero(self) -> None:
        close = np.full(60, 10.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = indicators.cci(close, close, close, window=20)
        assert np.all(out[19:] == 0.0)


class TestIchimoku:
    def test_lines_respect_the_window_extremes(self) -> None:
        h, lo, _, _ = _bars()
        out = indicators.ichimoku(h, lo)
        assert set(out) == {"tenkan", "kijun", "senkou_a", "senkou_b"}
        ok = np.isfinite(out["senkou_b"])
        assert ok.sum() == h.size - 51
        # Each line is the midpoint of its own trailing window's extremes.
        assert out["senkou_b"][-1] == pytest.approx(0.5 * (h[-52:].max() + lo[-52:].min()))
        assert out["kijun"][-1] == pytest.approx(0.5 * (h[-26:].max() + lo[-26:].min()))
        assert out["tenkan"][-1] == pytest.approx(0.5 * (h[-9:].max() + lo[-9:].min()))

    def test_short_series_rejected(self) -> None:
        with pytest.raises(ValueError, match="high needs >= 52"):
            indicators.ichimoku(np.ones(20), np.zeros(20))

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="match length"):
            indicators.ichimoku(np.arange(60.0), np.arange(59.0))


class TestDonchian:
    def test_bands_bracket_the_mid_line(self) -> None:
        h, lo, _, _ = _bars()
        out = indicators.donchian(h, lo, window=20)
        ok = np.isfinite(out["mid"])
        assert np.all(out["upper"][ok] >= out["mid"][ok])
        assert np.all(out["lower"][ok] <= out["mid"][ok])
        assert out["upper"][-1] == pytest.approx(h[-20:].max())


class TestKeltner:
    def test_half_width_is_the_atr_band(self) -> None:
        h, lo, c, _ = _bars()
        out = indicators.keltner(h, lo, c, window=20, mult=2.0)
        band = indicators.atr(h, lo, c, window=20)
        ok = np.isfinite(out["mid"])
        assert np.allclose(out["upper"][ok] - out["mid"][ok], 2.0 * band[ok])
        assert np.allclose(out["mid"][ok] - out["lower"][ok], 2.0 * band[ok])


class TestBollinger:
    def test_pct_b_and_bandwidth_shapes(self) -> None:
        _, _, c, _ = _bars()
        out = indicators.bollinger(c, window=20, num_sd=2.0)
        assert set(out) == {"mid", "upper", "lower", "pct_b", "bandwidth"}
        assert out["pct_b"].shape == c.shape
        assert np.isnan(out["pct_b"][:19]).all()

    def test_flat_series_has_no_percent_b(self) -> None:
        out = indicators.bollinger(np.full(40, 7.0), window=10)
        assert np.all(np.isnan(out["pct_b"][9:]))
        assert np.allclose(out["bandwidth"][9:], 0.0)

    @pytest.mark.parametrize("num_sd", [0.0, -1.0, float("nan")])
    def test_non_positive_sd_rejected(self, num_sd: float) -> None:
        with pytest.raises(ValueError, match="num_sd must be positive"):
            indicators.bollinger(np.arange(40.0), window=10, num_sd=num_sd)


class TestVolumeBlock:
    def test_obv_is_cumulative_signed_volume(self) -> None:
        close = np.arange(10.0, 30.0)
        volume = np.full(20, 3.0)
        out = indicators.obv(close, volume)
        # Bar 0 has no prior close, so sign(0) contributes nothing.
        assert np.allclose(out, 3.0 * np.arange(20))

    def test_obv_rejects_negative_volume(self) -> None:
        volume = np.ones(20)
        volume[0] = -1.0
        with pytest.raises(ValueError, match="volume must be finite"):
            indicators.obv(np.arange(20.0), volume)

    def test_ad_line_saturates_when_close_is_the_high(self) -> None:
        low = np.arange(20.0)
        high = low + 1.0
        close = high.copy()
        volume = np.full(20, 2.0)
        out = indicators.ad_line(high, low, close, volume)
        assert np.allclose(out, 2.0 * np.arange(1, 21))

    def test_ad_line_zero_range_contributes_nothing(self) -> None:
        close = np.full(20, 5.0)
        out = indicators.ad_line(close, close, close, np.ones(20))
        assert np.all(out == 0.0)

    def test_chaikin_money_flow_is_bounded(self) -> None:
        h, lo, c, v = _bars()
        out = indicators.chaikin_money_flow(h, lo, c, v, window=20)
        assert np.isfinite(out[19:]).all()
        assert np.all(np.abs(out[19:]) <= 1.0)
        assert np.isnan(out[:19]).all()

    def test_chaikin_money_flow_zero_volume_is_zero(self) -> None:
        h, lo, c, _ = _bars()
        out = indicators.chaikin_money_flow(h, lo, c, np.zeros(c.size), window=10)
        assert np.all(out[9:] == 0.0)

    def test_chaikin_oscillator_difference_of_emas(self) -> None:
        h, lo, c, v = _bars()
        out = indicators.chaikin_oscillator(h, lo, c, v, fast=3, slow=10)
        adl = indicators.ad_line(h, lo, c, v)
        assert np.allclose(
            out[np.isfinite(out)],
            (indicators.ema(adl, 3) - indicators.ema(adl, 10))[np.isfinite(out)],
        )

    def test_mfi_saturates_on_monotone_rising_prices(self) -> None:
        close = np.arange(20.0, 60.0)
        h = close + 0.5
        lo = close - 0.5
        out = indicators.mfi(h, lo, close, np.ones(40), window=14)
        assert np.all(out[14:] == 100.0)

    def test_force_index_flat_close_is_zero(self) -> None:
        out = indicators.force_index(np.full(40, 9.0), np.ones(40), window=13)
        assert np.allclose(out[12:], 0.0)

    def test_eom_is_finite_after_warmup(self) -> None:
        h, lo, _, v = _bars()
        out = indicators.eom(h, lo, v, window=14)
        assert np.isfinite(out[13:]).all()

    def test_vwap_of_constant_typical_price_is_that_price(self) -> None:
        close = np.full(30, 12.0)
        out = indicators.vwap(close, close, close, np.arange(1.0, 31.0))
        assert np.allclose(out, 12.0)

    def test_vwap_zero_volume_is_nan(self) -> None:
        close = np.full(10, 3.0)
        out = indicators.vwap(close, close, close, np.zeros(10))
        assert np.all(np.isnan(out))


class TestArmsIndex:
    def test_ratio_of_ratios(self) -> None:
        out = indicators.arms_index(
            np.array([2.0, 4.0]),
            np.array([1.0, 1.0]),
            np.array([2.0, 2.0]),
            np.array([1.0, 1.0]),
        )
        assert np.allclose(out, np.array([1.0, 2.0]))

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="match length"):
            indicators.arms_index(np.ones(3), np.ones(2), np.ones(3), np.ones(3))

    @pytest.mark.parametrize("bad", ["advances", "declines", "adv_volume", "dec_volume"])
    def test_non_positive_inputs_rejected(self, bad: str) -> None:
        kwargs = {
            "advances": np.ones(3),
            "declines": np.ones(3),
            "adv_volume": np.ones(3),
            "dec_volume": np.ones(3),
        }
        kwargs[bad] = np.array([1.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="must be positive"):
            indicators.arms_index(**kwargs)  # type: ignore[arg-type]


class TestChaikinVolatility:
    def test_constant_range_has_zero_roc(self) -> None:
        high = np.full(60, 11.0)
        low = np.full(60, 10.0)
        out = indicators.chaikin_volatility(high, low, window=10, roc_window=10)
        assert np.all(out[np.isfinite(out)] == 0.0)
        assert np.isnan(out[:19]).all()

    def test_widening_range_is_positive(self) -> None:
        low = np.zeros(80)
        high = np.linspace(1.0, 5.0, 80)
        out = indicators.chaikin_volatility(high, low, window=10, roc_window=10)
        assert out[-1] > 0.0

    def test_short_series_rejected(self) -> None:
        with pytest.raises(ValueError, match="high needs"):
            indicators.chaikin_volatility(np.ones(19), np.zeros(19), window=10, roc_window=10)


class TestMovingAverageFamily:
    def test_constant_series_survives_every_average(self) -> None:
        c = np.full(120, 8.0)
        for fn, kwargs in (
            (indicators.sma, {"window": 10}),
            (indicators.ema, {"window": 10}),
            (indicators.wma, {"window": 10}),
            (indicators.dema, {"window": 10}),
            (indicators.tema, {"window": 10}),
            (indicators.hma, {"window": 10}),
            (indicators.zlema, {"window": 10}),
        ):
            out = fn(c, **kwargs)
            assert out.shape == c.shape
            finite = out[np.isfinite(out)]
            assert np.allclose(finite, 8.0), fn.__name__

    def test_kama_tracks_a_smooth_trend(self) -> None:
        c = np.linspace(10.0, 30.0, 120)
        out = indicators.kama(c, window=10)
        assert np.isnan(out[:10]).all()
        assert np.isfinite(out[11:]).all()
        assert out[-1] > out[20]

    def test_kama_flat_series_has_zero_efficiency_ratio(self) -> None:
        with np.errstate(invalid="ignore", divide="ignore"):
            out = indicators.kama(np.full(60, 3.0), window=10)
        assert np.allclose(out[11:], 3.0)

    def test_kama_window_equal_to_length_is_index_error(self) -> None:
        with pytest.raises(IndexError):
            indicators.kama(np.arange(10.0), window=10)

    def test_window_larger_than_series_rejected(self) -> None:
        with pytest.raises(ValueError, match="needs >="):
            indicators.sma(np.arange(5.0), window=6)

    def test_series_shorter_than_window_rejected(self) -> None:
        with pytest.raises(ValueError, match="needs >="):
            indicators.ema(np.arange(5.0), window=6)


class TestVolatilityBandPrimitives:
    def test_true_range_of_a_constant_spread(self) -> None:
        h = np.full(20, 10.5)
        lo = np.full(20, 9.5)
        c = np.full(20, 10.0)
        assert np.allclose(indicators.true_range(h, lo, c), 1.0)

    def test_atr_and_natr_of_a_constant_spread(self) -> None:
        h = np.full(40, 11.0)
        lo = np.full(40, 10.0)
        c = np.full(40, 10.5)
        assert np.allclose(indicators.atr(h, lo, c, window=14)[13:], 1.0)
        assert np.allclose(indicators.natr(h, lo, c, window=14)[13:], 1.0 / 10.5)

    def test_invalid_ohlc_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid OHLC"):
            indicators.true_range(np.full(5, 9.0), np.full(5, 10.0), np.full(5, 9.5))

    def test_mismatched_ohlc_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            indicators.true_range(np.ones(5), np.zeros(5), np.ones(4))


class TestMomentumFamily:
    def test_rsi_saturates_both_ways_and_is_neutral_when_flat(self) -> None:
        up = indicators.rsi(np.linspace(10.0, 30.0, 60), window=14)
        down = indicators.rsi(np.linspace(30.0, 10.0, 60), window=14)
        flat = indicators.rsi(np.full(60, 5.0), window=14)
        assert up[-1] == pytest.approx(100.0)
        assert down[-1] == pytest.approx(0.0)
        assert flat[-1] == pytest.approx(50.0)

    def test_stochastic_flat_bars_sit_at_fifty(self) -> None:
        c = np.full(40, 7.0)
        out = indicators.stochastic(c, c, c, k_window=14, d_window=3)
        assert np.all(out["k"][13:] == 50.0)
        assert np.allclose(out["d"][15:], 50.0)

    def test_stochastic_high_close_pegs_k_at_one_hundred(self) -> None:
        c = np.linspace(10.0, 20.0, 40)
        out = indicators.stochastic(c, c - 1.0, c, k_window=14)
        assert out["k"][-1] == pytest.approx(100.0)

    def test_williams_r_flat_bars_sit_at_minus_fifty(self) -> None:
        c = np.full(40, 7.0)
        assert np.all(indicators.williams_r(c, c, c, window=14)[13:] == -50.0)

    def test_roc_and_momentum(self) -> None:
        c = np.array([1.0, 2.0, 4.0, 8.0])
        assert np.allclose(indicators.roc(c, window=2)[2:], [3.0, 3.0])
        assert np.allclose(indicators.momentum(c, window=1)[1:], [1.0, 2.0, 4.0])
        assert np.isnan(indicators.roc(c, window=2)[:2]).all()

    def test_cmo_flat_series_is_zero(self) -> None:
        with np.errstate(invalid="ignore", divide="ignore"):
            out = indicators.cmo(np.full(40, 4.0), window=14)
        assert np.all(out[14:] == 0.0)

    def test_cmo_rising_series_is_one_hundred(self) -> None:
        out = indicators.cmo(np.linspace(1.0, 40.0, 60), window=14)
        assert out[-1] == pytest.approx(100.0)

    def test_tsi_is_finite_after_warmup(self) -> None:
        _, _, c, _ = _bars()
        out = indicators.tsi(c, slow=25, fast=13)
        assert out.shape == c.shape
        assert np.isfinite(out[-1])

    def test_ultimate_oscillator_flat_bars_are_zero(self) -> None:
        c = np.full(60, 6.0)
        out = indicators.ultimate_oscillator(c, c, c)
        assert np.all(out[27:] == 0.0)

    def test_ultimate_oscillator_rising_bars_are_high(self) -> None:
        c = np.linspace(10.0, 40.0, 80)
        out = indicators.ultimate_oscillator(c + 0.5, c - 0.5, c)
        assert out[-1] >= 50.0


class TestCompositeOscillators:
    def test_fisher_flat_bars_are_zero(self) -> None:
        c = np.full(60, 3.0)
        out = indicators.fisher_transform(c, c, window=10)
        assert np.all(out["fisher"][9:] == 0.0)
        assert np.isnan(out["signal"][:10]).all()

    def test_elder_ray_offsets_from_the_ema(self) -> None:
        h = np.linspace(11.0, 20.0, 60)
        lo = h - 2.0
        c = h - 1.0
        out = indicators.elder_ray(h, lo, c, window=13)
        # Both powers are measured against the same EMA.
        assert np.allclose(h - out["bull_power"], lo - out["bear_power"], equal_nan=True)
        assert np.all(out["bull_power"][np.isfinite(out["bull_power"])] >= 0.0)

    def test_kst_warmup_and_shape(self) -> None:
        _, _, c, _ = _bars(60)
        out = indicators.kst(c)
        assert set(out) == {"kst", "signal"}
        assert np.isnan(out["kst"][:44]).all()

    def test_awesome_oscillator_of_flat_bars_is_zero(self) -> None:
        c = np.full(80, 5.0)
        out = indicators.awesome_oscillator(c, c)
        assert np.allclose(out[33:], 0.0)

    def test_awesome_oscillator_needs_the_slow_window(self) -> None:
        with pytest.raises(ValueError, match="needs >="):
            indicators.awesome_oscillator(np.ones(10), np.ones(10))


class TestStochRsi:
    def test_output_is_a_share_in_the_unit_interval(self) -> None:
        _, _, c, _ = _bars()
        out = indicators.stoch_rsi(c, rsi_window=14, stoch_window=14)
        finite = out[np.isfinite(out)]
        assert finite.size > 0
        assert np.all(finite >= 0.0)
        assert np.all(finite <= 1.0)

    def test_flat_prices_sit_at_the_midpoint(self) -> None:
        out = indicators.stoch_rsi(np.full(80, 4.0), rsi_window=14, stoch_window=14)
        finite = out[np.isfinite(out)]
        assert finite.size > 0
        assert np.allclose(finite, 0.5)
