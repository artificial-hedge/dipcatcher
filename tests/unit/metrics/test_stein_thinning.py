"""Tests for metrics/stein_thinning.py — kernel-Stein thinning/herding."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.stein_thinning import (
    bench_stein_thinning,
    compress_report,
    kernel_herding,
    median_bandwidth,
    mmd_rbf,
    rbf_gram,
    stein_kernel_gram,
    stein_thin,
    synth_posterior,
)


@pytest.fixture(scope="module")
def panel() -> tuple[np.ndarray, np.ndarray]:
    return synth_posterior(160, 4, 0.6, seed=0)


@pytest.fixture(scope="module")
def bw(panel: tuple[np.ndarray, np.ndarray]) -> float:
    return median_bandwidth(panel[0])


class TestGuards:
    def test_bad_x(self):
        with pytest.raises(ValueError):
            median_bandwidth(np.ones((1, 3)))
        with pytest.raises(ValueError):
            median_bandwidth(np.array([[np.inf, 0.0]]))
        with pytest.raises(ValueError):
            rbf_gram(np.ones((2, 2)), np.ones((2, 3)), 1.0)
        with pytest.raises(ValueError):
            rbf_gram(np.ones((2, 2)), np.ones((2, 2)), -1.0)
        with pytest.raises(ValueError):
            stein_thin(np.ones((10, 2)), np.ones((9, 2)), 5)
        with pytest.raises(ValueError):
            stein_thin(np.ones((10, 2)), np.ones((10, 2)), 0)
        with pytest.raises(ValueError):
            kernel_herding(np.ones((5, 2)), 6)
        with pytest.raises(ValueError):
            compress_report(np.ones((10, 2)), 10)
        with pytest.raises(ValueError):
            synth_posterior(4, 2, 0.5, 0)
        with pytest.raises(ValueError):
            synth_posterior(20, 2, 0.99, 0)


class TestKernelBasics:
    def test_gram_shape_symmetry(self, panel, bw):
        x, _ = panel
        k = rbf_gram(x[:10], x[:10], bw)
        assert k.shape == (10, 10)
        np.testing.assert_allclose(k, k.T)
        np.testing.assert_allclose(np.diag(k), 1.0)
        assert (k <= 1.0 + 1e-12).all() and (k > 0).all()

    def test_median_bandwidth_positive(self, bw):
        assert bw > 0 and np.isfinite(bw)

    def test_median_bandwidth_scales(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((60, 3))
        bw1 = median_bandwidth(x)
        bw2 = median_bandwidth(10.0 * x)
        assert bw2 / bw1 == pytest.approx(10.0)

    def test_mmd_self_zero(self, panel, bw):
        x, _ = panel
        assert mmd_rbf(x, x, bw) == pytest.approx(0.0, abs=1e-10)

    def test_mmd_shifted_positive(self, bw):
        rng = np.random.default_rng(2)
        a = rng.standard_normal((50, 2))
        b = rng.standard_normal((50, 2)) + 3.0
        assert mmd_rbf(a, b, bw) > 0.5


class TestHerding:
    def test_herding_unique_m(self, panel, bw):
        x, _ = panel
        idx = kernel_herding(x, 32, bw)
        assert idx.size == 32 and np.unique(idx).size == 32
        assert idx.min() >= 0 and idx.max() < x.shape[0]

    def test_herding_deterministic(self, panel, bw):
        x, _ = panel
        np.testing.assert_array_equal(kernel_herding(x, 20, bw), kernel_herding(x, 20, bw))

    def test_herding_beats_random(self, panel, bw):
        x, _ = panel
        rng = np.random.default_rng(0)
        m = 32
        herd = kernel_herding(x, m, bw)
        rand = rng.choice(x.shape[0], size=m, replace=False)
        assert mmd_rbf(x[herd], x, bw) < mmd_rbf(x[rand], x, bw)


class TestSteinKernel:
    def test_gram_finite_symmetric(self, panel, bw):
        x, s = panel
        k0 = stein_kernel_gram(x, s, bw)
        assert k0.shape == (x.shape[0], x.shape[0])
        np.testing.assert_allclose(k0, k0.T, atol=1e-10)
        assert np.isfinite(k0).all()

    def test_stein_gram_sensible_scale(self, panel, bw):
        # true-score samples have modest KSD² (kernel mean bounded)
        x, s = panel
        k0 = stein_kernel_gram(x, s, bw)
        assert abs(k0.mean()) < 1.0


class TestSteinThin:
    def test_thin_size_and_bounds(self, panel, bw):
        x, s = panel
        idx = stein_thin(x, s, 32, bw)
        assert idx.size == 32
        assert np.unique(idx).size == 32
        assert idx.min() >= 0 and idx.max() < x.shape[0]

    def test_thin_deterministic(self, panel, bw):
        x, s = panel
        np.testing.assert_array_equal(stein_thin(x, s, 24, bw), stein_thin(x, s, 24, bw))

    def test_thin_m1(self, panel, bw):
        x, s = panel
        assert stein_thin(x, s, 1, bw).size == 1

    def test_thin_lowers_ksd_vs_random(self, panel, bw):
        x, s = panel
        rng = np.random.default_rng(5)
        thin = stein_thin(x, s, 32, bw)
        rand = rng.choice(x.shape[0], size=32, replace=False)
        k0 = stein_kernel_gram(x, s, bw)
        ksd_thin = k0[np.ix_(thin, thin)].mean()
        ksd_rand = k0[np.ix_(rand, rand)].mean()
        assert ksd_thin < ksd_rand

    def test_thin_subset_has_lower_mmd(self, panel, bw):
        x, s = panel
        rng = np.random.default_rng(3)
        thin = stein_thin(x, s, 32, bw)
        rand = rng.choice(x.shape[0], size=32, replace=False)
        assert mmd_rbf(x[thin], x, bw) < mmd_rbf(x[rand], x, bw)


class TestCompressReport:
    def test_report_keys(self, panel, bw):
        x, _ = panel
        rep = compress_report(x, 32, bw, seed=0)
        assert set(rep) == {"mmd_herd", "mmd_random", "mmd_edge"}
        assert rep["mmd_edge"] < 1.0


class TestSynth:
    def test_synth_shapes_score(self):
        x, s = synth_posterior(100, 5, 0.4, 7)
        assert x.shape == (100, 5) and s.shape == (100, 5)
        # score of N(0,Σ) is -Σ^{-1} x: ||score + Σ^{-1}x|| = 0
        cov = np.full((5, 5), 0.4) + np.eye(5) * 0.6
        resid = s + np.linalg.solve(cov, x.T).T
        assert np.abs(resid).max() < 1e-10

    def test_synth_deterministic(self):
        a, _ = synth_posterior(50, 3, 0.3, 11)
        b, _ = synth_posterior(50, 3, 0.3, 11)
        np.testing.assert_array_equal(a, b)


class TestBench:
    def test_bench_keys_finite(self):
        blob = bench_stein_thinning(seed=0)
        assert blob
        for k, v in blob.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float) and np.isfinite(v)

    def test_bench_contract_no_forbidden(self):
        forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
        for k in bench_stein_thinning(seed=0):
            assert not forbidden.intersection(k.split("_"))

    def test_bench_science(self):
        blob = bench_stein_thinning(seed=0)
        assert blob["synthetic_thin_mmd_edge"] < 1.0
        assert blob["synthetic_herd_edge"] < 1.0
        assert blob["synthetic_ksd_thin_vs_random"] < 1.0
        assert blob["synthetic_mmd_self"] == pytest.approx(0.0, abs=1e-10)
        assert blob["synthetic_determinism"] == 1.0
