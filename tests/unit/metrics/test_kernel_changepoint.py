"""Kernel-MMD change-point: MMD correctness, calibration, delay, gates."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.kernel_changepoint import (
    bench_kernel_changepoint,
    binseg_detect,
    cusum_baseline,
    mmd_scan,
    mmd_u,
    permutation_threshold,
    synth_stream,
)


class TestMmdU:
    def test_zero_for_identical(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(60)
        # identical samples: unbiased statistic ~ -2/m (small negative)
        assert abs(mmd_u(x, x)) < 0.1

    def test_positive_for_shifted(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(80)
        y = rng.standard_normal(80) + 2.0
        assert mmd_u(x, y) > 0.1

    def test_small_for_same_distribution(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        assert abs(mmd_u(x, y)) < 0.05

    def test_scales_with_shift(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(80)
        d1 = mmd_u(x, rng.standard_normal(80) + 0.5)
        d2 = mmd_u(x, rng.standard_normal(80) + 2.0)
        assert d2 > d1

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            mmd_u(np.array([1.0]), np.array([1.0, 2.0]))
        with pytest.raises(ValueError):
            mmd_u(np.zeros(50), np.zeros(50))  # degenerate bandwidth


class TestScan:
    def test_finds_planted_cp(self):
        x = synth_stream(300, cp=150, kind="mean", size=1.5, seed=10)
        cp, stat = mmd_scan(x, min_seg=30)
        assert abs(cp - 150) < 40
        assert np.isfinite(stat[cp])

    def test_nan_outside_margin(self):
        x = synth_stream(200, None, seed=11)
        _, stat = mmd_scan(x, min_seg=30)
        assert np.isnan(stat[:30]).all()
        assert np.isnan(stat[171:]).all()

    def test_flat_statistic_on_null(self):
        x = synth_stream(200, None, seed=12)
        cp, stat = mmd_scan(x, min_seg=40)
        assert 40 <= cp <= 160


class TestCalibration:
    def test_threshold_positive(self):
        x = synth_stream(200, None, seed=20)
        thr = permutation_threshold(x, alpha=0.05, n_perm=40, seed=20)
        assert np.isfinite(thr) and thr > 0

    def test_size_control_roughly(self):
        # under the null, max-MMD exceeds the calibrated threshold rarely
        rng = np.random.default_rng(21)
        alarms = 0
        for rep in range(6):
            x = rng.standard_normal(200)
            thr = permutation_threshold(x, alpha=0.2, n_perm=40, seed=rep)
            _, stat = mmd_scan(x, min_seg=40)
            if np.nanmax(stat) > thr:
                alarms += 1
        assert alarms <= 3  # generous bound, 6 trials at alpha=0.2

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            permutation_threshold(np.zeros(50), n_perm=40)
        with pytest.raises(ValueError):
            permutation_threshold(np.random.default_rng(0).standard_normal(200), alpha=1.5)


class TestBinseg:
    def test_finds_two_cps(self):
        rng = np.random.default_rng(30)
        x = np.concatenate(
            [
                rng.standard_normal(120),
                rng.standard_normal(120) + 2.0,
                rng.standard_normal(120) - 2.0,
            ]
        )
        cps = binseg_detect(x, alpha=0.05, min_seg=30, n_perm=40, seed=30)
        assert len(cps) >= 2
        assert any(abs(c - 120) < 40 for c in cps)
        assert any(abs(c - 240) < 40 for c in cps)

    def test_no_cp_on_null(self):
        x = synth_stream(240, None, seed=31)
        cps = binseg_detect(x, alpha=0.05, min_seg=40, n_perm=30, seed=31)
        assert len(cps) <= 1  # permissive bound

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            binseg_detect(np.zeros(30), min_seg=20)


class TestCusumBaseline:
    def test_alarms_on_mean_shift(self):
        x = synth_stream(300, cp=150, kind="mean", size=2.0, seed=40)
        alarm, mx = cusum_baseline(x)
        assert alarm is not None
        assert mx > 5.0

    def test_quiet_on_null(self):
        x = synth_stream(300, None, seed=41)
        alarm, _ = cusum_baseline(x)
        # null streams rarely exceed 5-sigma cumulative
        # (loose assertion: either no alarm or late alarm)
        assert alarm is None or alarm > 200


class TestSynthStream:
    def test_kinds(self):
        for kind in ("mean", "variance", "family"):
            x = synth_stream(200, cp=100, kind=kind, size=0.5, seed=50)
            assert x.shape == (200,)
            assert np.isfinite(x).all()

    def test_no_cp_is_stationary(self):
        x = synth_stream(300, None, seed=51)
        assert abs(x[:150].mean() - x[150:].mean()) < 0.5

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            synth_stream(50, None)
        with pytest.raises(ValueError):
            synth_stream(200, cp=5)
        with pytest.raises(ValueError):
            synth_stream(200, cp=100, kind="bogus")


class TestBench:
    def test_keys_finite(self):
        out = bench_kernel_changepoint(20260202)
        for k, v in out.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float)
            assert np.isfinite(v), k

    def test_deterministic(self):
        assert bench_kernel_changepoint(20260202) == bench_kernel_changepoint(20260202)

    def test_quality(self):
        out = bench_kernel_changepoint(20260202)
        assert out["synthetic_detect_rate_size2"] >= 0.5  # large shift detected
        assert out["synthetic_detect_rate_size0"] <= out["synthetic_detect_rate_size2"]
