"""features/cycles.py — spectral filters, cycle estimators, and argument guards.

Inputs are SYNTHETIC sinusoids so each expected period is known analytically.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.features import cycles


def _sine(period: float, n: int = 128, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2.0 * math.pi * t / period) + 0.05 * rng.normal(size=n)


class TestGoertzelPower:
    def test_power_peaks_at_the_planted_period(self) -> None:
        x = _sine(8)
        powers = [cycles.goertzel_power(x, period) for period in (6, 8, 10)]
        assert powers[1] == max(powers)
        assert powers[1] > 0.0

    @pytest.mark.parametrize("period", [3, 2, 0, -1, 100000])
    def test_out_of_range_period_rejected(self, period: int) -> None:
        with pytest.raises(ValueError, match="period must be"):
            cycles.goertzel_power(_sine(8), period)

    @pytest.mark.parametrize("period", [8.0, "8", True])
    def test_non_integer_period_rejected(self, period: object) -> None:
        with pytest.raises(ValueError, match="period must be an integer"):
            cycles.goertzel_power(_sine(8), period)  # type: ignore[arg-type]

    def test_period_beyond_half_the_series_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"<= len\(x\)/2"):
            cycles.goertzel_power(_sine(8, n=32), 20)

    def test_short_or_nonfinite_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite vector"):
            cycles.goertzel_power(np.arange(15.0), 6)
        bad = _sine(8)
        bad[3] = np.nan
        with pytest.raises(ValueError, match="finite vector"):
            cycles.goertzel_power(bad, 8)


class TestCyclePeriodogram:
    def test_default_grid_covers_the_admissible_band(self) -> None:
        periods, power = cycles.cycle_periodogram(_sine(8, n=64))
        assert periods[0] == 6.0
        assert periods[-1] == 32.0
        assert power.shape == periods.shape
        assert periods[int(np.argmax(power))] == pytest.approx(8.0, abs=1.0)

    def test_explicit_grid_is_echoed(self) -> None:
        periods, power = cycles.cycle_periodogram(_sine(8), np.array([8, 12, 16]))
        assert list(periods) == [8.0, 12.0, 16.0]
        assert periods[int(np.argmax(power))] == pytest.approx(8.0)

    @pytest.mark.parametrize("grid", [[8], [3, 8], [8, 999]])
    def test_invalid_grid_rejected(self, grid: list[int]) -> None:
        with pytest.raises(ValueError, match=r"integers in \[4, len\(x\)/2\]"):
            cycles.cycle_periodogram(_sine(8), np.array(grid))


class TestDominantCycleFft:
    def test_recovers_the_planted_period(self) -> None:
        out = cycles.dominant_cycle_fft(_sine(10, n=256), min_period=6, max_period=20)
        assert 9.0 < out["period"] < 11.0
        assert 0.0 < out["power_fraction"] <= 1.0
        assert out["snr_vs_mean"] >= 1.0

    def test_flat_series_has_no_power_fraction(self) -> None:
        # Zero total power: the fraction falls back to 0 and the snr is 0/0.
        with np.errstate(invalid="ignore", divide="ignore"):
            out = cycles.dominant_cycle_fft(np.ones(128))
        assert out["power_fraction"] == 0.0
        assert np.isnan(out["snr_vs_mean"])

    @pytest.mark.parametrize(
        ("lo", "hi"),
        [(3, 20), (6, 200), (20, 20), (30, 20)],
    )
    def test_invalid_band_rejected(self, lo: int, hi: int) -> None:
        with pytest.raises(ValueError, match="require 4 <="):
            cycles.dominant_cycle_fft(_sine(10, n=128), min_period=lo, max_period=hi)


class TestEhlersQuadrature:
    def test_loop_path_skips_warmup_and_propagates_nan(self) -> None:
        x = np.arange(12.0)
        x[7] = np.nan
        in_phase, quad = cycles._ehlers_quadrature(x)
        assert np.isnan(in_phase[:9]).all()
        assert np.isnan(quad[:9]).all()

    def test_vectorized_path_matches_the_loop_on_clean_input(self) -> None:
        x = np.linspace(-1.0, 1.0, 20) ** 2
        in_phase, quad = cycles._ehlers_quadrature(x)
        smooth = np.convolve(x, np.array([4.0, 3.0, 2.0, 1.0]) / 10.0, mode="valid")
        expected_quad = (
            0.0962 * smooth[6:]
            + 0.5769 * smooth[4:-2]
            - 0.5769 * smooth[2:-4]
            - 0.0962 * smooth[:-6]
        )
        assert np.allclose(quad[9:], expected_quad)
        assert np.allclose(in_phase[9:], smooth[3:-3][: quad[9:].size])


class TestHilbertInstantaneousFrequency:
    def test_estimate_tracks_the_planted_period(self) -> None:
        fast = cycles.hilbert_instantaneous_frequency(_sine(12, n=240))
        slow = cycles.hilbert_instantaneous_frequency(_sine(40, n=240))
        fast_med = float(np.median(fast[np.isfinite(fast)]))
        slow_med = float(np.median(slow[np.isfinite(slow)]))
        assert np.isfinite(fast_med) and np.isfinite(slow_med)
        assert fast_med < slow_med
        assert np.isfinite(fast).sum() > 50

    def test_smooth_one_returns_the_raw_series(self) -> None:
        raw = cycles.hilbert_instantaneous_frequency(_sine(20, n=200), smooth=1)
        smoothed = cycles.hilbert_instantaneous_frequency(_sine(20, n=200), smooth=5)
        assert np.isfinite(raw).sum() > np.isfinite(smoothed).sum()
        assert np.isnan(raw[:9]).all()

    @pytest.mark.parametrize("smooth", [0, -3, True, 2.5])
    def test_invalid_smooth_rejected(self, smooth: object) -> None:
        with pytest.raises(ValueError, match="smooth must be a positive integer"):
            cycles.hilbert_instantaneous_frequency(_sine(20, n=64), smooth=smooth)  # type: ignore[arg-type]


class TestSupersmoother:
    def test_constant_series_is_unchanged(self) -> None:
        out = cycles.supersmoother(np.full(32, 3.5))
        assert np.allclose(out, 3.5)

    def test_output_shape_matches_input(self) -> None:
        x = _sine(10)
        assert cycles.supersmoother(x).shape == x.shape

    def test_output_lags_a_step_input(self) -> None:
        x = np.concatenate([np.zeros(32), np.ones(32)])
        out = cycles.supersmoother(x)
        assert out[31] == pytest.approx(0.0)
        assert out[40] > 0.0
        assert out[-1] == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.parametrize("period", [1.0, 0.0, -5.0, float("nan")])
    def test_invalid_period_rejected(self, period: float) -> None:
        with pytest.raises(ValueError, match="period must be >= 2"):
            cycles.supersmoother(_sine(10), period=period)


class TestRoofingFilter:
    def test_constant_series_is_removed_by_the_highpass(self) -> None:
        out = cycles.roofing_filter(np.full(80, 2.0))
        assert np.allclose(out[20:], 0.0, atol=1e-6)

    @pytest.mark.parametrize("hp", [2.0, 1.0, 0.0, float("inf")])
    def test_highpass_period_guard(self, hp: float) -> None:
        with pytest.raises(ValueError, match="hp_period must be > 2"):
            cycles.roofing_filter(_sine(10, n=64), hp_period=hp)

    @pytest.mark.parametrize("lp", [1.0, 0.0, -2.0])
    def test_lowpass_period_guard(self, lp: float) -> None:
        with pytest.raises(ValueError, match="lp_period must be >= 2"):
            cycles.roofing_filter(_sine(10, n=64), lp_period=lp)


class TestBandpassFilter:
    def test_leading_bars_are_zero_and_output_is_bounded(self) -> None:
        out = cycles.bandpass_filter(_sine(20, n=200))
        assert np.all(out[:3] == 0.0)
        assert np.isfinite(out).all()

    def test_narrow_bandwidth_still_runs(self) -> None:
        out = cycles.bandpass_filter(_sine(20, n=200), bandwidth=0.05)
        assert np.isfinite(out).all()

    @pytest.mark.parametrize("period", [1.0, 0.0, float("nan")])
    def test_invalid_period_rejected(self, period: float) -> None:
        with pytest.raises(ValueError, match="period must be >= 2"):
            cycles.bandpass_filter(_sine(20, n=64), period=period)

    @pytest.mark.parametrize("bandwidth", [0.0, 1.0, -0.1, 1.5, float("nan")])
    def test_invalid_bandwidth_rejected(self, bandwidth: float) -> None:
        with pytest.raises(ValueError, match=r"bandwidth must be in \(0, 1\)"):
            cycles.bandpass_filter(_sine(20, n=64), bandwidth=bandwidth)


class TestHilbertTransformIndicator:
    def test_amplitude_is_the_quadrature_hypot(self) -> None:
        out = cycles.hilbert_transform_indicator(_sine(12, n=120))
        ok = np.isfinite(out["in_phase"]) & np.isfinite(out["quadrature"])
        assert np.allclose(
            out["amplitude"][ok], np.hypot(out["in_phase"][ok], out["quadrature"][ok])
        )
        assert np.all(np.abs(out["phase"][ok]) <= math.pi)
        assert np.isnan(out["amplitude"][:9]).all()
