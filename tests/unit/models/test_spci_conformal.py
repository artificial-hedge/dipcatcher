"""SPCI: residual-quantile forecasting, coverage tracking, fail-closed."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.spci_conformal import (
    aci_comparison,
    bench_spci_conformal,
    lag_features,
    pinball_ridge_fit,
    pinball_ridge_predict,
    spci_walk_forward,
    synth_drift_series,
)


class TestPinballRidge:
    def test_recovers_median(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((400, 3))
        y = 1.0 + x @ np.array([0.5, -0.3, 0.8]) + rng.standard_normal(400) * 0.1
        w = pinball_ridge_fit(x, y, alpha=0.5)
        pred = pinball_ridge_predict(w, x)
        assert np.mean(np.abs(y - pred)) < 0.25

    def test_quantile_direction(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((400, 2))
        y = x[:, 0] + rng.standard_normal(400)
        w10 = pinball_ridge_fit(x, y, alpha=0.1)
        w90 = pinball_ridge_fit(x, y, alpha=0.9)
        p10 = pinball_ridge_predict(w10, x)
        p90 = pinball_ridge_predict(w90, x)
        assert np.all(p10 < p90)
        assert np.mean(y < p90) > np.mean(y < p10)

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            pinball_ridge_fit(np.ones((4, 2)), np.ones(4), alpha=0.5)
        with pytest.raises(ValueError):
            pinball_ridge_fit(np.ones((20, 2)), np.ones(19), alpha=0.5)
        with pytest.raises(ValueError):
            pinball_ridge_fit(np.ones((20, 2)), np.ones(20), alpha=1.5)


class TestLagFeatures:
    def test_content(self):
        s = np.arange(10.0)
        x = lag_features(s, n_lags=3)
        assert x.shape == (7, 3)
        np.testing.assert_allclose(x[0], [2.0, 1.0, 0.0])
        np.testing.assert_allclose(x[-1], [8.0, 7.0, 6.0])

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            lag_features(np.arange(5.0), n_lags=6)


class TestWalkForward:
    def test_intervals_cover(self):
        y, yhat = synth_drift_series(800, shift_size=1.5, seed=2)
        path = spci_walk_forward(y, yhat, alpha=0.1, min_train=80)
        valid = ~np.isnan(path.covered)
        assert valid.sum() > 500
        assert path.coverage > 0.7  # rough nominal ~0.9 on synthetic drift

    def test_widths_positive(self):
        y, yhat = synth_drift_series(600, seed=3)
        path = spci_walk_forward(y, yhat, alpha=0.1, min_train=60)
        v = ~np.isnan(path.qhat)
        assert (path.upper[v] > path.lower[v]).all()

    def test_width_adapts_to_shift(self):
        y, yhat = synth_drift_series(1200, shift_at=0.5, shift_size=3.0, seed=4)
        path = spci_walk_forward(y, yhat, alpha=0.1, min_train=80)
        q = path.qhat[~np.isnan(path.qhat)]
        # quantile forecast after the vol shift exceeds the pre-shift level
        early = np.nanmean(q[:300])
        late = np.nanmean(q[-200:])
        assert late > early

    def test_deterministic(self):
        y, yhat = synth_drift_series(500, seed=5)
        a = spci_walk_forward(y, yhat, alpha=0.1, min_train=60)
        b = spci_walk_forward(y, yhat, alpha=0.1, min_train=60)
        np.testing.assert_allclose(np.nan_to_num(a.qhat), np.nan_to_num(b.qhat))

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            spci_walk_forward(np.arange(20.0), np.arange(20.0), min_train=60)
        with pytest.raises(ValueError):
            spci_walk_forward(np.arange(200.0), np.arange(199.0), min_train=60)
        with pytest.raises(ValueError):
            spci_walk_forward(np.arange(200.0), np.arange(200.0), alpha=0.9)


class TestComparison:
    def test_aci_coverage_err_finite(self):
        y, yhat = synth_drift_series(800, seed=6)
        err = aci_comparison(y, yhat, alpha=0.1)
        assert np.isfinite(err)
        assert err >= 0.0

    def test_aci_recovers_post_shift(self):
        y, yhat = synth_drift_series(2000, shift_size=3.0, seed=7)
        err = aci_comparison(y, yhat, alpha=0.1)
        assert err < 0.25  # ACI adapts back toward 90%


class TestSyntheticSeries:
    def test_shift_visible(self):
        y, _ = synth_drift_series(1000, shift_at=0.5, shift_size=3.0, seed=8)
        assert np.std(y[600:]) > np.std(y[:400])

    def test_deterministic(self):
        a = synth_drift_series(200, seed=9)
        b = synth_drift_series(200, seed=9)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])


class TestBench:
    def test_keys_finite(self):
        out = bench_spci_conformal(20260131)
        for k, v in out.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float)
            assert np.isfinite(v), k

    def test_deterministic(self):
        assert bench_spci_conformal(20260131) == bench_spci_conformal(20260131)

    def test_quality(self):
        out = bench_spci_conformal(20260131)
        assert out["synthetic_spci_coverage_post"] > 0.75
        assert out["synthetic_determinism"] == 1.0
