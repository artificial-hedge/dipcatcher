"""Fractional OU: spectrum, spectral simulation, Whittle estimation."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.fractional_ou import (
    bench_fractional_ou,
    estimate_fou,
    fou_autocov_theoretical,
    fou_spectrum,
    simulate_fou_exact,
    synth_fou_observed,
    whittle_loglik,
)


class TestSpectrum:
    def test_positive(self):
        w = np.linspace(0.01, 5.0, 50)
        s = fou_spectrum(w, 0.4, 0.5, 1.0)
        assert np.all(s > 0)

    def test_ou_limit(self):
        # H=1/2: fOU spectrum should match OU Lorentzian shape
        w = np.linspace(0.01, 5.0, 50)
        s = fou_spectrum(w, 0.5, 0.5, 1.0)
        s_norm = s / s[0]
        lorentz = 1.0 / (0.25 + w**2)
        lorentz_norm = lorentz / lorentz[0]
        np.testing.assert_allclose(s_norm, lorentz_norm, rtol=0.1)

    def test_low_freq_rougher(self):
        # f(w) ~ |w|^{1-2H}/(a^2+w^2): lower H -> more high-frequency mass
        w = np.array([0.05, 1.0])
        s_rough = fou_spectrum(w, 0.3, 0.5)
        s_smooth = fou_spectrum(w, 0.7, 0.5)
        ratio_r = s_rough[0] / s_rough[1]
        ratio_s = s_smooth[0] / s_smooth[1]
        assert ratio_r < ratio_s

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            fou_spectrum(np.array([1.0]), 1.2, 0.5)
        with pytest.raises(ValueError):
            fou_spectrum(np.array([1.0]), 0.4, -0.5)
        with pytest.raises(ValueError):
            fou_spectrum(np.array([1.0]), 0.4, 0.5, 0.0)


class TestSimulation:
    def test_shape_demeaned(self):
        x = simulate_fou_exact(512, 0.4, 0.5, seed=1)
        assert x.shape == (512,)
        assert abs(x.mean()) < 1e-8

    def test_deterministic(self):
        a = simulate_fou_exact(256, 0.4, 0.5, seed=2)
        b = simulate_fou_exact(256, 0.4, 0.5, seed=2)
        np.testing.assert_allclose(a, b)

    def test_rougher_has_higher_short_lag_var(self):
        # lower H gives bigger short-lag increment variance relative to level var
        x_r = simulate_fou_exact(2048, 0.3, 0.5, seed=3)
        x_s = simulate_fou_exact(2048, 0.7, 0.5, seed=3)
        inc_r = np.var(np.diff(x_r)) / np.var(x_r)
        inc_s = np.var(np.diff(x_s)) / np.var(x_s)
        assert inc_r > inc_s

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            simulate_fou_exact(16, 0.4, 0.5)
        with pytest.raises(ValueError):
            simulate_fou_exact(256, 1.1, 0.5)


class TestEstimation:
    def test_h_recovery(self):
        x = synth_fou_observed(2048, 0.4, 0.5, seed=4)
        est = estimate_fou(x, seed=4)
        assert abs(est.h - 0.4) < 0.2
        assert est.converged

    def test_h_grid(self):
        for h in (0.3, 0.6):
            x = synth_fou_observed(2048, h, 0.5, seed=5)
            est = estimate_fou(x, seed=5)
            assert abs(est.h - h) < 0.25

    def test_whittle_prefers_truth(self):
        x = synth_fou_observed(1024, 0.4, 0.5, seed=6)
        ll_true = whittle_loglik(x, 0.4, 0.5, 1.0)
        ll_wrong = whittle_loglik(x, 0.8, 0.5, 1.0)
        assert ll_true > ll_wrong

    def test_deterministic(self):
        x = synth_fou_observed(512, 0.4, 0.5, seed=7)
        e1 = estimate_fou(x, seed=7)
        e2 = estimate_fou(x, seed=7)
        assert e1.h == e2.h

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            estimate_fou(np.arange(10.0))
        with pytest.raises(ValueError):
            whittle_loglik(np.arange(10.0), 0.4, 0.5, 1.0)


class TestAutocov:
    def test_theoretical_decreasing(self):
        lags = np.arange(0, 10, dtype=float)
        ac = fou_autocov_theoretical(lags, 0.4, 0.8)
        assert ac[0] > ac[-1]
        assert ac[0] > 0


class TestBench:
    def test_keys_finite(self):
        out = bench_fractional_ou(20260201)
        for k, v in out.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float)
            assert np.isfinite(v), k

    def test_deterministic(self):
        assert bench_fractional_ou(20260201) == bench_fractional_ou(20260201)

    def test_quality(self):
        out = bench_fractional_ou(20260201)
        assert out["synthetic_h_err_mean"] < 0.3
        assert out["synthetic_determinism"] == 1.0
