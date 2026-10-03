"""EM multivariate-Hawkes estimation: responsibilities, recovery, fail-closed."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.hawkes_em import (
    bench_hawkes_em,
    branching_spectral_radius,
    em_hawkes,
    em_vs_mle_bivariate,
    simulate_hawkes_mv,
    synth_bivariate_hawkes,
)


class TestSimulate:
    def test_streams_sorted_nonempty(self):
        mu = np.array([0.4, 0.3])
        a = np.array([[0.4, 0.1], [0.2, 0.3]])
        b = np.array([[1.0, 1.0], [1.0, 1.0]])
        s = simulate_hawkes_mv(mu, a, b, horizon=200.0, seed=1)
        assert len(s) == 2
        for stream in s:
            assert stream.size > 10
            assert np.all(np.diff(stream) > 0)

    def test_excitation_adds_events(self):
        mu = np.array([0.3, 0.3])
        b = np.ones((2, 2))
        quiet = simulate_hawkes_mv(mu, np.zeros((2, 2)), b, 300.0, seed=2)
        hot = simulate_hawkes_mv(mu, np.array([[0.6, 0.3], [0.3, 0.6]]), b, 300.0, seed=2)
        assert sum(s.size for s in hot) > sum(s.size for s in quiet)

    def test_explosive_rejected(self):
        with pytest.raises(ValueError):
            simulate_hawkes_mv(
                np.array([0.3]),
                np.array([[2.0]]),
                np.array([[1.0]]),
                horizon=100.0,
            )

    def test_deterministic(self):
        mu = np.array([0.4])
        a = np.array([[0.4]])
        b = np.array([[1.0]])
        x = simulate_hawkes_mv(mu, a, b, 100.0, seed=3)
        y = simulate_hawkes_mv(mu, a, b, 100.0, seed=3)
        np.testing.assert_allclose(x[0], y[0])

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            simulate_hawkes_mv(np.array([-0.1]), np.array([[0.1]]), np.array([[1.0]]), 50.0)
        with pytest.raises(ValueError):
            simulate_hawkes_mv(np.array([0.1]), np.array([[0.1]]), np.array([[1.0]]), -1.0)


class TestEM:
    def test_loglik_nondecreasing(self):
        streams, *_ = synth_bivariate_hawkes(seed=4, horizon=300.0)
        fit = em_hawkes(streams, n_iter=30)
        d = np.diff(fit.loglik_trace)
        assert np.mean(d >= -1e-6) > 0.8  # EM ascent (EM bound on curvature)

    def test_branching_recovery(self):
        streams, mu, alpha, beta = synth_bivariate_hawkes(seed=5, horizon=600.0)
        fit = em_hawkes(streams, n_iter=60)
        true_b = alpha / beta
        err = np.linalg.norm(fit.branching - true_b) / np.linalg.norm(true_b)
        assert err < 0.8

    def test_mu_recovery(self):
        streams, mu, alpha, beta = synth_bivariate_hawkes(seed=6, horizon=600.0)
        fit = em_hawkes(streams, n_iter=60)
        assert np.linalg.norm(fit.mu - mu) / np.linalg.norm(mu) < 0.8

    def test_stability_classification(self):
        streams, *_ = synth_bivariate_hawkes(seed=7, horizon=400.0)
        fit = em_hawkes(streams, n_iter=40)
        assert fit.spectral_radius < 1.0

    def test_excitation_share_bounds(self):
        streams, *_ = synth_bivariate_hawkes(seed=8, horizon=400.0)
        fit = em_hawkes(streams, n_iter=40)
        assert np.all(fit.excitation_share >= 0.0)
        assert np.all(fit.excitation_share <= 1.0)

    def test_deterministic(self):
        streams, *_ = synth_bivariate_hawkes(seed=9, horizon=200.0)
        f1 = em_hawkes(streams, n_iter=20)
        f2 = em_hawkes(streams, n_iter=20)
        np.testing.assert_allclose(f1.alpha, f2.alpha)

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            em_hawkes([np.array([1.0, 2.0])], n_iter=5)  # too few events
        with pytest.raises(ValueError):
            em_hawkes([np.array([3.0] * 20)], n_iter=5)  # non-increasing
        with pytest.raises(ValueError):
            em_hawkes([], n_iter=5)


class TestBranching:
    def test_spectral_radius(self):
        a = np.array([[0.5, 0.0], [0.0, 0.25]])
        b = np.ones((2, 2))
        assert abs(branching_spectral_radius(a, b) - 0.5) < 1e-9

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            branching_spectral_radius(np.ones((2, 2)), np.zeros((2, 2)))
        with pytest.raises(ValueError):
            branching_spectral_radius(np.ones(3), np.ones(3))


class TestComparison:
    def test_em_vs_mle_finite(self):
        streams, *_ = synth_bivariate_hawkes(seed=10, horizon=500.0)
        out = em_vs_mle_bivariate(streams, n_iter=40)
        for k, v in out.items():
            assert np.isfinite(v), k

    def test_wrong_arity(self):
        streams, *_ = synth_bivariate_hawkes(seed=11, horizon=200.0)
        with pytest.raises(ValueError):
            em_vs_mle_bivariate(streams[:1])


class TestBench:
    def test_keys_finite(self):
        out = bench_hawkes_em(20260201)
        for k, v in out.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float)
            assert np.isfinite(v), k

    def test_deterministic(self):
        assert bench_hawkes_em(20260201) == bench_hawkes_em(20260201)

    def test_quality(self):
        out = bench_hawkes_em(20260201)
        assert out["synthetic_determinism"] == 1.0
        assert out["synthetic_stability_correct"] == 1.0
