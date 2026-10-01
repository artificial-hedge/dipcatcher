"""Brock-Hommes HAM: skeleton, sorting, stylized facts, fail-closed."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.heterogeneous_abm import (
    bench_heterogeneous_abm,
    deterministic_skeleton,
    dominance_grid,
    lyapunov_exponent,
    simulate_brock_hommes,
    stylized_facts,
)


class TestSimulate:
    def test_shape_and_bounds(self):
        sim = simulate_brock_hommes(500, beta=50.0, seed=1)
        assert sim.prices.shape == (500,)
        assert sim.returns.shape == (499,)
        assert sim.fractions.shape == (500, 2)
        assert np.all(sim.fractions >= 0.0) and np.all(sim.fractions <= 1.0)
        np.testing.assert_allclose(sim.fractions.sum(axis=1), 1.0, atol=1e-9)

    def test_zero_beta_half_half(self):
        # beta=0: logit fractions stay exactly 50/50
        sim = simulate_brock_hommes(500, beta=0.0, seed=2)
        np.testing.assert_allclose(sim.fractions, 0.5, atol=1e-9)

    def test_high_beta_sharpens(self):
        # strong trend rule: high-beta population locks into one regime
        sim = simulate_brock_hommes(1200, beta=400.0, h=4.0, cost_chartist=0.0, seed=3)
        assert sim.fractions[200:].max(axis=1).mean() > 0.95

    def test_deterministic(self):
        a = simulate_brock_hommes(400, beta=60.0, seed=4)
        b = simulate_brock_hommes(400, beta=60.0, seed=4)
        np.testing.assert_allclose(a.prices, b.prices)
        np.testing.assert_allclose(a.fractions, b.fractions)

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            simulate_brock_hommes(50, beta=10.0)
        with pytest.raises(ValueError):
            simulate_brock_hommes(500, beta=-1.0)
        with pytest.raises(ValueError):
            simulate_brock_hommes(500, beta=10.0, g=0.0)
        with pytest.raises(ValueError):
            simulate_brock_hommes(500, beta=10.0, rho=1.0)
        with pytest.raises(ValueError):
            simulate_brock_hommes(500, beta=10.0, lam=0.0)


class TestSkeleton:
    def test_low_beta_converges(self):
        sk = deterministic_skeleton(1500, beta=0.0)
        assert np.std(sk.prices[-200:]) < 1e-6  # settles to fixed point

    def test_deterministic(self):
        a = deterministic_skeleton(500, beta=100.0)
        b = deterministic_skeleton(500, beta=100.0)
        np.testing.assert_allclose(a.prices, b.prices)


class TestDiagnostics:
    def test_stylized_facts_keys(self):
        rng = np.random.default_rng(5)
        r = rng.standard_normal(1000) * 0.01
        f = stylized_facts(r)
        assert "excess_kurtosis" in f
        assert "abs_autocorr_lag1" in f
        assert abs(f["return_autocorr_lag1"]) < 0.15  # iid has ~0 autocorr

    def test_fat_tails_flag(self):
        rng = np.random.default_rng(6)
        r = np.concatenate([rng.standard_normal(900) * 0.01, rng.standard_t(3, 100) * 0.01])
        f = stylized_facts(r)
        assert f["excess_kurtosis"] > 0.5

    def test_lyapunov_finite(self):
        lam = lyapunov_exponent(800, beta=100.0)
        assert np.isfinite(lam)

    def test_dominance_grid(self):
        bs, dom = dominance_grid(np.array([0.0, 100.0, 500.0]), n=1200)
        assert dom.shape == (3,)
        assert np.all(np.isfinite(dom))
        assert dom[-1] > dom[0]  # sorting sharpens with beta

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            stylized_facts(np.arange(50.0))
        with pytest.raises(ValueError):
            dominance_grid(np.array([1.0]), n=300)
        with pytest.raises(ValueError):
            dominance_grid(np.array([-1.0, 2.0]), n=300)
        with pytest.raises(ValueError):
            lyapunov_exponent(800, beta=-1.0)


class TestBench:
    def test_keys_finite(self):
        out = bench_heterogeneous_abm(20260201)
        for k, v in out.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float)
            assert np.isfinite(v), k

    def test_deterministic(self):
        assert bench_heterogeneous_abm(20260201) == bench_heterogeneous_abm(20260201)

    def test_quality(self):
        out = bench_heterogeneous_abm(20260201)
        assert out["synthetic_skeleton_conv_std"] < 1e-6
        assert out["synthetic_sorting_sharpens"] == 1.0
        assert out["synthetic_determinism"] == 1.0
