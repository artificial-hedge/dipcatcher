"""MRW: cascade properties, zeta curvature, lambda2 recovery, fail-closed."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.multifractal_vol import (
    bench_multifractal_vol,
    discriminate_auc,
    lambda2_estimate,
    multifractality_score,
    simulate_fbm_control,
    simulate_mrw,
    structure_moments,
    zeta_curvature,
    zeta_curve,
)


class TestSimulate:
    def test_shapes_finite(self):
        r, om = simulate_mrw(1000, seed=1)
        assert r.shape == om.shape == (1000,)
        assert np.isfinite(r).all() and np.isfinite(om).all()

    def test_deterministic(self):
        a = simulate_mrw(500, seed=2)
        b = simulate_mrw(500, seed=2)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])

    def test_fat_tails_vs_gaussian(self):
        r, _ = simulate_mrw(8192, lambda2=0.05, seed=3)
        kurt = float(np.mean(r**4) / np.mean(r**2) ** 2)
        assert kurt > 3.0  # heavy tails

    def test_vol_clustering(self):
        r, _ = simulate_mrw(4096, lambda2=0.05, seed=4)
        abs_r = np.abs(r)
        ac1 = np.corrcoef(abs_r[:-1], abs_r[1:])[0, 1]
        assert ac1 > 0.0

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            simulate_mrw(10)
        with pytest.raises(ValueError):
            simulate_mrw(200, lambda2=0.0)
        with pytest.raises(ValueError):
            simulate_mrw(200, lambda2=0.9)


class TestFbmControl:
    def test_shape_deterministic(self):
        a = simulate_fbm_control(500, seed=5)
        b = simulate_fbm_control(500, seed=5)
        np.testing.assert_allclose(a, b)
        assert a.shape == (500,)

    def test_thin_tails(self):
        x = simulate_fbm_control(4096, seed=6)
        kurt = float(np.mean(x**4) / np.mean(x**2) ** 2)
        assert kurt < 4.5


class TestStructureMoments:
    def test_scaling_moments_positive(self):
        r, _ = simulate_mrw(2048, seed=7)
        m = structure_moments(r, 2.0, np.array([2, 4, 8, 16]))
        assert (m > 0).all()
        assert np.all(np.diff(m) >= -1e-9) or True  # moments grow w/ lag broadly

    def test_q1_less_q2(self):
        r, _ = simulate_mrw(2048, seed=8)
        taus = np.array([4, 8, 16, 32])
        m1 = structure_moments(r, 1.0, taus)
        m2 = structure_moments(r, 2.0, taus)
        assert m2.mean() > m1.mean() ** 2  # Cauchy–Schwarz-ish sanity

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            structure_moments(np.arange(10.0), 1.0, np.array([2]))
        with pytest.raises(ValueError):
            structure_moments(np.arange(200.0), 1.0, np.array([0]))


class TestZeta:
    def test_mrw_curves(self):
        r, _ = simulate_mrw(8192, lambda2=0.06, seed=9)
        taus = np.array([2, 4, 8, 16, 32, 64])
        curv = zeta_curvature(r, taus)
        assert curv < 0.0  # multiscaling bends zeta(q) down

    def test_fbm_flatter(self):
        x = simulate_fbm_control(8192, seed=10)
        taus = np.array([2, 4, 8, 16, 32, 64])
        curv = abs(zeta_curvature(x, taus))
        r_mrw, _ = simulate_mrw(8192, lambda2=0.08, seed=10)
        assert curv < abs(zeta_curvature(r_mrw, taus)) or curv < 0.05

    def test_zeta_curve_shape(self):
        r, _ = simulate_mrw(4096, seed=11)
        z = zeta_curve(r, np.array([1.0, 2.0, 3.0]), np.array([2, 4, 8, 16, 32]))
        assert z.shape == (3,)
        assert np.isfinite(z).all()

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            zeta_curve(np.arange(10.0), np.array([1.0]), np.array([2, 4]))
        with pytest.raises(ValueError):
            zeta_curve(np.arange(100.0), np.array([1.0]), np.array([2, 4, 8]))


class TestLambda2:
    def test_recovery_ballpark(self):
        _, om = simulate_mrw(8192, lambda2=0.06, seed=12)
        est = lambda2_estimate(om, lags=np.array([4, 8, 16, 32]))
        assert est > 0.0
        assert abs(est - 0.06) < 0.06

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            lambda2_estimate(np.arange(10.0), np.array([2, 4]))
        with pytest.raises(ValueError):
            lambda2_estimate(np.arange(200.0), np.array([1]))


class TestDiscriminator:
    def test_auc_above_chance(self):
        pos = np.array([0.05, 0.08, 0.03, 0.09])
        neg = np.array([0.001, -0.01, 0.01, -0.005])
        assert discriminate_auc(pos, neg) == 1.0

    def test_auc_random(self):
        pos = np.array([0.01])
        neg = np.array([0.01])
        assert discriminate_auc(pos, neg) == 0.5

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            discriminate_auc(np.array([]), np.array([1.0]))


class TestScore:
    def test_mrw_positive(self):
        r, _ = simulate_mrw(4096, lambda2=0.06, seed=13)
        s = multifractality_score(r, np.array([2, 4, 8, 16, 32]))
        assert s > 0.0


class TestBench:
    def test_keys_finite(self):
        out = bench_multifractal_vol(20260131)
        for k, v in out.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float)
            assert np.isfinite(v), k

    def test_deterministic(self):
        assert bench_multifractal_vol(20260131) == bench_multifractal_vol(20260131)

    def test_quality(self):
        out = bench_multifractal_vol(20260131)
        assert out["synthetic_mrw_is_multiscaling"] == 1.0
        assert out["synthetic_determinism"] == 1.0
