"""Neural TPP: Hawkes MLE recovery, fallback parity, torch gating, fail-closed."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.neural_tpp import (
    bench_neural_tpp,
    evaluate_fallback,
    fit_marked_fallback,
    hawkes_intensity_path,
    hawkes_mle_univariate,
    poisson_loglik,
    simulate_marked_hawkes,
    torch_available,
)

HAS_TORCH = torch_available()


def _sim(seed: int = 0, n: int = 400):
    return simulate_marked_hawkes(n, mu=0.8, alpha=0.5, beta=1.2, n_marks=2, seed=seed)


class TestSimulation:
    def test_counts_and_order(self):
        t, k = _sim(seed=1)
        assert t.size == 400
        assert np.all(np.diff(t) > 0)
        assert set(np.unique(k)) <= {0, 1}

    def test_deterministic(self):
        a = simulate_marked_hawkes(100, 0.8, 0.5, 1.2, seed=2)
        b = simulate_marked_hawkes(100, 0.8, 0.5, 1.2, seed=2)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_mark_probs_respected(self):
        t, k = simulate_marked_hawkes(
            2000,
            0.8,
            0.5,
            1.2,
            n_marks=3,
            mark_probs=np.array([0.7, 0.2, 0.1]),
            seed=3,
        )
        freq = np.bincount(k, minlength=3) / k.size
        assert abs(freq[0] - 0.7) < 0.08

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            simulate_marked_hawkes(3, 0.8, 0.5, 1.2)
        with pytest.raises(ValueError):
            simulate_marked_hawkes(50, -1.0, 0.5, 1.2)
        with pytest.raises(ValueError):
            simulate_marked_hawkes(50, 0.8, 1.5, 1.0)  # supercritical


class TestHawkesMLE:
    def test_recovers_mu(self):
        t, _ = _sim(seed=4, n=1200)
        fit = hawkes_mle_univariate(t)
        assert abs(fit["mu"] - 0.8) / 0.8 < 0.5

    def test_recovers_branching(self):
        t, _ = _sim(seed=5, n=1500)
        fit = hawkes_mle_univariate(t)
        assert abs(fit["branching_ratio"] - 0.5 / 1.2) < 0.3

    def test_beats_poisson_loglik(self):
        t, _ = _sim(seed=6, n=800)
        fit = hawkes_mle_univariate(t)
        assert fit["loglik"] > poisson_loglik(t)

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            hawkes_mle_univariate(np.arange(5.0))
        with pytest.raises(ValueError):
            hawkes_mle_univariate(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))


class TestIntensityPath:
    def test_shape_and_floor(self):
        t, k = _sim(seed=7)
        lam = hawkes_intensity_path(t, k, mark=0, mu=0.5, alpha=0.3, beta=1.0)
        assert lam.size == int((k == 0).sum())
        assert (lam >= 0.5).all()

    def test_fail_closed(self):
        t, k = _sim(seed=8)
        with pytest.raises(ValueError):
            hawkes_intensity_path(t, k, mark=0, mu=-1.0, alpha=0.3, beta=1.0)


class TestFallback:
    def test_result_keys(self):
        t, k = _sim(seed=9, n=300)
        out = fit_marked_fallback(t, k, n_marks=2)
        for key in ("loglik", "mu_mean", "alpha_mean", "beta_mean", "branching_ratio_mean"):
            assert key in out
            assert np.isfinite(out[key])

    def test_evaluate_keys(self):
        t, k = _sim(seed=10, n=300)
        out = evaluate_fallback(t, k, n_marks=2)
        assert np.isfinite(out["loglik_eval"])
        assert 0.0 <= out["mark_accuracy"] <= 1.0

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            fit_marked_fallback(np.arange(3.0), np.array([0, 0, 0]), n_marks=2)
        with pytest.raises(ValueError):
            fit_marked_fallback(np.arange(10.0), np.array([0] * 9 + [5]), n_marks=2)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (nn extra)")
class TestNeuralPath:
    def test_fit_returns(self):
        from quant_fund.models.neural_tpp import fit_neural_tpp

        t, k = _sim(seed=11, n=150)
        fit = fit_neural_tpp(t, k, n_marks=2, steps=20)
        assert np.isfinite(fit.loglik_eval)
        assert 0.0 <= fit.mark_accuracy <= 1.0

    def test_deterministic_seed(self):
        from quant_fund.models.neural_tpp import fit_neural_tpp

        t, k = _sim(seed=12, n=100)
        a = fit_neural_tpp(t, k, n_marks=2, steps=10, seed=13)
        b = fit_neural_tpp(t, k, n_marks=2, steps=10, seed=13)
        assert a.loglik_eval == b.loglik_eval


class TestBench:
    def test_keys_finite(self):
        out = bench_neural_tpp(20260131)
        for key, v in out.items():
            assert key.startswith("synthetic_")
            assert isinstance(v, float)
            assert np.isfinite(v), key

    def test_deterministic(self):
        assert bench_neural_tpp(20260131) == bench_neural_tpp(20260131)

    def test_fallback_quality(self):
        out = bench_neural_tpp(20260131)
        assert out["synthetic_hawkes_mu_relerr"] < 0.5
        assert out["synthetic_determinism"] == 1.0
