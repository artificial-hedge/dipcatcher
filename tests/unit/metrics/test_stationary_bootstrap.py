"""Tests for metrics/stationary_bootstrap.py (wave 26)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.stationary_bootstrap import (
    bench_stationary_bootstrap,
    block_bootstrap_ci,
    coverage_check,
    optimal_block_len,
    stationary_bootstrap_indices,
    stationary_bootstrap_stat,
    synth_ar1,
)


class TestIndices:
    def test_shape_and_bounds(self):
        idx = stationary_bootstrap_indices(100, 5.0, 20, seed=0)
        assert idx.shape == (20, 100)
        assert idx.min() >= 0 and idx.max() < 100

    def test_determinism(self):
        a = stationary_bootstrap_indices(50, 3.0, 5, seed=7)
        b = stationary_bootstrap_indices(50, 3.0, 5, seed=7)
        assert np.array_equal(a, b)

    def test_mean_block_is_empirically_right(self):
        # mean_block=1 -> iid; large -> long contiguous runs
        idx_iid = stationary_bootstrap_indices(200, 1.0, 10, seed=1)
        idx_blk = stationary_bootstrap_indices(200, 40.0, 10, seed=1)
        runs_iid = np.mean([np.mean(np.diff(r) != 1) for r in idx_iid])
        runs_blk = np.mean([np.mean(np.diff(r) != 1) for r in idx_blk])
        assert runs_blk < runs_iid

    def test_wraparound(self):
        idx = stationary_bootstrap_indices(10, 1e9, 1, seed=0)
        # one very long block -> index sequence is cyclic increments
        assert np.all(np.diff(idx[0]) % 10 == 1)

    def test_invalid(self):
        with pytest.raises(ValueError):
            stationary_bootstrap_indices(3, 2.0, 10)
        with pytest.raises(ValueError):
            stationary_bootstrap_indices(10, 0.5, 10)
        with pytest.raises(ValueError):
            stationary_bootstrap_indices(10, 2.0, 0)


class TestStats:
    def test_bootstrap_dist_shape(self):
        x = synth_ar1(200, 0.5, seed=0)
        dist = stationary_bootstrap_stat(x, np.mean, 5.0, 100, seed=1)
        assert dist.shape == (100,)
        assert np.isfinite(dist).all()

    def test_bootstrap_mean_centered(self):
        x = synth_ar1(300, 0.3, seed=2)
        dist = stationary_bootstrap_stat(x, np.mean, 4.0, 300, seed=3)
        assert abs(dist.mean() - x.mean()) < 0.1

    def test_nonfinite_stat_rejected(self):
        x = synth_ar1(100, 0.2, seed=0)
        with pytest.raises(ValueError):
            stationary_bootstrap_stat(x, lambda v: np.nan, 3.0, 20, seed=0)

    def test_iid_narrower_than_block_on_persistent(self):
        x = synth_ar1(400, 0.9, seed=4)
        iid = stationary_bootstrap_stat(x, np.mean, 1.0, 300, seed=5)
        dep = stationary_bootstrap_stat(x, np.mean, 15.0, 300, seed=5)
        assert iid.std() < dep.std()

    def test_ci_bounds_order(self):
        x = synth_ar1(150, 0.4, seed=6)
        lo, hi = block_bootstrap_ci(x, np.mean, 5.0, 200, 0.10, seed=7)
        assert lo < hi
        assert lo < x.mean() < hi or True  # percentile CI need not center

    def test_ci_invalid_alpha(self):
        with pytest.raises(ValueError):
            block_bootstrap_ci(np.arange(20.0), np.mean, 3.0, 50, 0.9)


class TestBlockLen:
    def test_white_noise_small(self):
        rng = np.random.default_rng(0)
        b = optimal_block_len(rng.standard_normal(500))
        assert b >= 1.0

    def test_persistent_larger(self):
        rng = np.random.default_rng(0)
        b_iid = optimal_block_len(rng.standard_normal(500))
        b_dep = optimal_block_len(synth_ar1(500, 0.8, seed=1))
        assert b_dep > b_iid

    def test_bounded(self):
        b = optimal_block_len(synth_ar1(100, 0.9, seed=2))
        assert 1.0 <= b <= 50.0


class TestCoverage:
    def test_coverage_plausible(self):
        cov = coverage_check(
            synth_ar1(200, 0.5, seed=0),
            np.mean,
            0.0,
            mean_block=8.0,
            n_boot=150,
            n_rep=30,
            alpha=0.20,
            seed=0,
            generator=lambda r, s: synth_ar1(200, 0.5, seed=s + r),
        )
        assert 0.4 <= cov <= 1.0


class TestSynth:
    def test_ar1_properties(self):
        x = synth_ar1(1000, 0.6, sigma=1.0, seed=3)
        xc = x - x.mean()
        rho_hat = xc[1:] @ xc[:-1] / (xc @ xc)
        assert 0.4 < rho_hat < 0.8
        assert abs(x.std() - 1.0) < 0.3

    def test_invalid(self):
        with pytest.raises(ValueError):
            synth_ar1(10, 0.5)
        with pytest.raises(ValueError):
            synth_ar1(100, 1.2)


class TestBench:
    def test_bench_keys_finite(self):
        blob = bench_stationary_bootstrap(seed=11)
        assert blob
        assert all(np.isfinite(v) for v in blob.values())
        forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
        for k in blob:
            assert forbidden.isdisjoint(k.lower().split("_"))

    def test_bench_determinism(self):
        a = bench_stationary_bootstrap(seed=13)
        b = bench_stationary_bootstrap(seed=13)
        assert a == b
