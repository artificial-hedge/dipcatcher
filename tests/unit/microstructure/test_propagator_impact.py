"""Propagator impact: kernel estimation, diagnostics, cost curve, fail-closed."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.propagator_impact import (
    bench_propagator_impact,
    expected_cost_curve,
    fit_kernel_ls,
    fit_power_law,
    kernel_diagnostics,
    power_law_kernel,
    propagator_response,
    simulate_impacted_series,
)


def _sim(seed: int = 0, n: int = 20_000, n_lags: int = 30):
    lags = np.arange(n_lags, dtype=float)
    g = power_law_kernel(lags, c=1.0, beta=0.5)
    flow, resp = simulate_impacted_series(n, g, persistence=0.5, seed=seed)
    return lags, g, flow, resp


class TestResponse:
    def test_causal_convolution(self):
        flow = np.array([1.0, 0.0, 0.0, 0.0])
        g = np.array([2.0, 1.0, 0.5])
        out = propagator_response(flow, g)
        np.testing.assert_allclose(out, [2.0, 1.0, 0.5, 0.0])

    def test_zero_kernel_zero_response(self):
        flow = np.random.default_rng(0).standard_normal(50)
        np.testing.assert_allclose(propagator_response(flow, np.zeros(3)), 0.0)

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            propagator_response(np.array([1.0, 2.0]), np.array([1.0]))
        with pytest.raises(ValueError):
            propagator_response(np.arange(10.0), np.array([-1.0, 0.5]))
        with pytest.raises(ValueError):
            propagator_response(np.full(10, np.nan), np.ones(2))


class TestPowerLawKernel:
    def test_value(self):
        g = power_law_kernel(np.array([0.0, 1.0, 3.0]), c=2.0, beta=0.5, l0=1.0)
        np.testing.assert_allclose(g, [2.0, 2.0 / np.sqrt(2), 1.0])

    def test_monotone_decay(self):
        lags = np.arange(30, dtype=float)
        g = power_law_kernel(lags, c=1.0, beta=0.6)
        assert np.all(np.diff(g) <= 0)

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            power_law_kernel(np.arange(5.0), c=0.0, beta=0.5)
        with pytest.raises(ValueError):
            power_law_kernel(np.arange(5.0), c=1.0, beta=-0.5)


class TestKernelEstimation:
    def test_recovery_on_known_kernel(self):
        lags, g_true, flow, resp = _sim(seed=1)
        g_hat = fit_kernel_ls(flow, resp, n_lags=30)
        rel = np.linalg.norm(g_hat - g_true) / np.linalg.norm(g_true)
        assert rel < 0.35

    def test_nonneg_kernel(self):
        _, _, flow, resp = _sim(seed=2)
        g_hat = fit_kernel_ls(flow, resp, n_lags=20)
        assert (g_hat >= 0).all()

    def test_shape(self):
        _, _, flow, resp = _sim(seed=3)
        assert fit_kernel_ls(flow, resp, n_lags=15).shape == (15,)

    def test_fail_closed(self):
        flow, resp = np.arange(50.0), np.arange(50.0)
        with pytest.raises(ValueError):
            fit_kernel_ls(flow, resp[:40], n_lags=5)
        with pytest.raises(ValueError):
            fit_kernel_ls(flow, resp, n_lags=0)
        with pytest.raises(ValueError):
            fit_kernel_ls(flow, resp, n_lags=100)


class TestPowerLawFit:
    def test_recovers_beta(self):
        lags = np.arange(1.0, 50.0)
        g = power_law_kernel(lags, c=1.5, beta=0.45)
        c, beta = fit_power_law(lags, g)
        assert abs(beta - 0.45) < 0.02
        assert abs(c - 1.5) < 0.2

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            fit_power_law(np.arange(5.0), np.zeros(5))
        with pytest.raises(ValueError):
            fit_power_law(np.arange(5.0), np.arange(4.0))


class TestDiagnostics:
    def test_halflife_known(self):
        lags = np.arange(60, dtype=float)
        g = power_law_kernel(lags, c=1.0, beta=0.5)
        diag = kernel_diagnostics(lags, g)
        # G(l) = (1+l)^-0.5 halves at l = 3
        assert abs(diag.half_life - 3.0) < 0.6
        assert abs(diag.beta - 0.5) < 0.05

    def test_transient_share_bounds(self):
        lags = np.arange(40, dtype=float)
        g = power_law_kernel(lags, c=1.0, beta=0.4)
        diag = kernel_diagnostics(lags, g)
        assert 0.0 < diag.transient_share <= 1.0
        assert diag.permanent_mass > 0

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            kernel_diagnostics(np.arange(5.0), np.zeros(5))
        with pytest.raises(ValueError):
            kernel_diagnostics(np.arange(4.0), np.array([1.0, 0.5]))


class TestCostCurve:
    def test_instant_gt_uniform(self):
        lags = np.arange(30, dtype=float)
        g = power_law_kernel(lags, c=1.0, beta=0.5)
        curve = expected_cost_curve(g, horizon=20)
        assert curve[0] > curve[-1]

    def test_monotone_decay(self):
        lags = np.arange(30, dtype=float)
        g = power_law_kernel(lags, c=1.0, beta=0.6)
        curve = expected_cost_curve(g, horizon=15)
        # cost per unit declines as spread lengthens
        assert np.mean(np.diff(curve) <= 0) > 0.9

    def test_flat_kernel_flat_curve(self):
        # G = const (fully permanent): spreading doesn't help
        g = np.full(20, 1.0)
        curve = expected_cost_curve(g, horizon=10)
        # uniform cost ~ (G0 + 2 sum (t-l))/t^2 → ~1 as t grows
        assert abs(curve[-1] - 1.0) < 0.15

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            expected_cost_curve(np.ones(5), horizon=0)


class TestSimulation:
    def test_deterministic(self):
        f1, r1 = simulate_impacted_series(300, np.array([1.0, 0.5]), seed=7)
        f2, r2 = simulate_impacted_series(300, np.array([1.0, 0.5]), seed=7)
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_allclose(r1, r2)

    def test_response_consistent(self):
        g = np.array([1.0, 0.6, 0.3])
        flow, resp = simulate_impacted_series(500, g, seed=8)
        np.testing.assert_allclose(resp, propagator_response(flow, g))

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            simulate_impacted_series(5, np.ones(2), seed=0)
        with pytest.raises(ValueError):
            simulate_impacted_series(50, np.ones(2), sigma_flow=0.0)
        with pytest.raises(ValueError):
            simulate_impacted_series(50, np.ones(2), persistence=1.5)


class TestBench:
    def test_keys_finite(self):
        out = bench_propagator_impact(20260131)
        for k, v in out.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float)
            assert np.isfinite(v), k

    def test_deterministic(self):
        assert bench_propagator_impact(20260131) == bench_propagator_impact(20260131)

    def test_quality(self):
        out = bench_propagator_impact(20260131)
        assert out["synthetic_kernel_recovery_relerr"] < 0.4
        assert out["synthetic_beta_err"] < 0.2
        assert out["synthetic_cost_monotone_frac"] > 0.8
        assert out["synthetic_determinism"] == 1.0
