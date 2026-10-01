"""Tests for metrics/knockoffs.py — SYNTHETIC correctness only."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.knockoffs import (
    bench_knockoffs,
    knockoff_select,
    knockoff_stat_diff,
    knockoff_threshold,
    sample_gaussian_knockoffs,
    stability_select,
    synth_linear,
)


class TestSampling:
    def test_shape_and_determinism(self) -> None:
        x, _y, _t = synth_linear(n=300, p=20, k=4, seed=0)
        k1 = sample_gaussian_knockoffs(x, seed=1)
        k2 = sample_gaussian_knockoffs(x, seed=1)
        k3 = sample_gaussian_knockoffs(x, seed=2)
        assert k1.shape == x.shape
        assert np.array_equal(k1, k2)
        assert not np.array_equal(k1, k3)

    def test_exchangeability_moments(self) -> None:
        x, _y, _t = synth_linear(n=400, p=15, k=3, seed=2)
        xk = sample_gaussian_knockoffs(x, seed=0)
        # knockoffs share the mean vector and the diagonal of Sigma
        assert np.abs(x.mean(axis=0) - xk.mean(axis=0)).max() < 0.5
        diag_x = np.diag(np.cov(x, rowvar=False))
        diag_k = np.diag(np.cov(xk, rowvar=False))
        assert np.abs(diag_x - diag_k).max() / diag_x.max() < 0.5

    def test_sdp_feasible(self) -> None:
        x, _y, _t = synth_linear(n=300, p=12, k=3, seed=3)
        xk = sample_gaussian_knockoffs(x, method="sdp", seed=0)
        assert xk.shape == x.shape

    def test_invalid(self) -> None:
        x = np.zeros((50, 10))
        with pytest.raises(ValueError):
            sample_gaussian_knockoffs(x, method="bogus")
        with pytest.raises(ValueError):
            sample_gaussian_knockoffs(np.ones((5, 20)))


class TestThreshold:
    def test_selects_when_clear(self) -> None:
        w = np.array([3.0, 2.5, 2.0, -0.5, 0.4, -0.3, 0.2, -0.1])
        tau = knockoff_threshold(w, q=0.5)
        assert np.isfinite(tau)
        assert tau <= 2.0

    def test_inf_when_too_noisy(self) -> None:
        rng = np.random.default_rng(0)
        w = rng.standard_normal(50) * 0.1
        tau = knockoff_threshold(w, q=0.05)
        assert np.isinf(tau)

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            knockoff_threshold(np.array([0.1, 0.2]), q=0.9)
        with pytest.raises(ValueError):
            knockoff_threshold(np.array([]), q=0.1)


class TestSelect:
    def test_recovers_strong_signals(self) -> None:
        x, y, true_idx = synth_linear(n=600, p=40, k=8, rho=0.1, amplitude=5.0, seed=4)
        sel = knockoff_select(x, y, q=0.2, stat="omp", seed=0)
        inter = np.intersect1d(sel, true_idx)
        assert inter.size >= 6

    def test_null_selects_rarely(self) -> None:
        x, _y, _t = synth_linear(n=400, p=20, k=4, seed=5)
        rng = np.random.default_rng(6)
        y = rng.standard_normal(400)
        sel = knockoff_select(x, y, q=0.1, seed=0)
        assert sel.size <= 20  # mostly empty on pure null

    def test_determinism(self) -> None:
        x, y, _ = synth_linear(n=300, p=20, k=4, seed=7)
        s1 = knockoff_select(x, y, q=0.2, seed=0)
        s2 = knockoff_select(x, y, q=0.2, seed=0)
        assert np.array_equal(s1, s2)

    def test_invalid(self) -> None:
        x, y, _ = synth_linear(n=300, p=20, k=4, seed=8)
        with pytest.raises(ValueError):
            knockoff_select(x, y, q=0.8)
        with pytest.raises(ValueError):
            knockoff_stat_diff(x, y[:-1], x)


class TestStability:
    def test_frequencies_bounds(self) -> None:
        x, y, true_idx = synth_linear(n=500, p=30, k=5, amplitude=4.0, seed=9)
        res = stability_select(x, y, n_boot=10, q=0.3, seed=0)
        assert res.inclusion_freq.shape == (30,)
        assert np.all(res.inclusion_freq >= 0) and np.all(res.inclusion_freq <= 1)
        # true features should out-rank nulls in inclusion frequency
        assert res.inclusion_freq[true_idx].mean() > np.delete(res.inclusion_freq, true_idx).mean()


class TestBench:
    def test_bench_keys_finite(self) -> None:
        blob = bench_knockoffs()
        assert len(blob) >= 10
        assert all(np.isfinite(v) for v in blob.values())
        for key in blob:
            assert key.startswith("synthetic_")
            assert {"sharpe", "sortino", "calmar", "pnl", "nav"}.isdisjoint(key.lower().split("_"))

    def test_bench_reports_tradeoff(self) -> None:
        blob = bench_knockoffs()
        assert blob["synthetic_omp_power"] > 0.7
        assert blob["synthetic_ridge_fdr"] <= 0.25
        assert blob["synthetic_null_reject_rate"] <= 0.4
        assert blob["synthetic_determinism"] == 1.0
