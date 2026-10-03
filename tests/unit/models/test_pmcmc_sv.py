"""PMCMC-SV: PF loglik, MH sanity, posterior recovery, fail-closed."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.pmcmc_sv import (
    bench_pmcmc_sv,
    pmmh_sv,
    simulate_sv,
    sv_pf_loglik,
)


class TestSimulate:
    def test_shapes_and_finite(self):
        r, h = simulate_sv(300, seed=1)
        assert r.shape == h.shape == (300,)
        assert np.isfinite(r).all() and np.isfinite(h).all()

    def test_deterministic(self):
        a = simulate_sv(200, seed=2)
        b = simulate_sv(200, seed=2)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])

    def test_vol_clustering(self):
        r, _ = simulate_sv(3000, phi=0.97, sigma_eta=0.2, seed=3)
        abs_r = np.abs(r)
        ac1 = np.corrcoef(abs_r[:-1], abs_r[1:])[0, 1]
        assert ac1 > 0.05  # SV produces positive |r| autocorrelation

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            simulate_sv(5)
        with pytest.raises(ValueError):
            simulate_sv(50, phi=1.2)
        with pytest.raises(ValueError):
            simulate_sv(50, sigma_eta=0.0)


class TestPFLoglik:
    def test_finite(self):
        r, _ = simulate_sv(300, seed=4)
        ll = sv_pf_loglik(r, -4.0, 0.95, 0.15, n_particles=100, seed=0)
        assert np.isfinite(ll)

    def test_better_params_better_loglik(self):
        r, _ = simulate_sv(400, mu=-4.0, phi=0.95, sigma_eta=0.15, seed=5)
        good = sv_pf_loglik(r, -4.0, 0.95, 0.15, n_particles=200, seed=0)
        bad = sv_pf_loglik(r, 0.5, 0.0, 2.0, n_particles=200, seed=0)
        assert good > bad

    def test_stability_with_particles(self):
        r, _ = simulate_sv(300, seed=6)
        l_small = sv_pf_loglik(r, -4.0, 0.95, 0.15, n_particles=100, seed=0)
        l_big = sv_pf_loglik(r, -4.0, 0.95, 0.15, n_particles=800, seed=0)
        # same seed: estimates should be close (few-sigma of PF noise)
        assert abs(l_small - l_big) < 15.0

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            sv_pf_loglik(np.arange(5.0), -4.0, 0.9, 0.2)
        with pytest.raises(ValueError):
            sv_pf_loglik(np.random.default_rng(0).standard_normal(50), -4.0, 1.5, 0.2)


class TestPMMH:
    def test_runs_and_shapes(self):
        r, _ = simulate_sv(300, seed=7)
        res = pmmh_sv(r, n_iter=80, n_particles=80, seed=0)
        assert res.post_mean.shape == (3,)
        assert res.chain_loglik.shape == (80,)
        assert res.effective_sizes.shape == (3,)

    def test_acceptance_reasonable(self):
        r, _ = simulate_sv(300, seed=8)
        res = pmmh_sv(r, n_iter=120, n_particles=100, seed=0)
        assert 0.0 <= res.acceptance_rate <= 1.0
        assert res.acceptance_rate > 0.02  # not a dead chain

    def test_posterior_in_ballpark(self):
        r, _ = simulate_sv(500, mu=-4.0, phi=0.95, sigma_eta=0.15, seed=9)
        res = pmmh_sv(r, n_iter=200, n_particles=150, seed=0)
        assert abs(res.post_mean[0] - (-4.0)) < 1.5
        assert 0.5 < res.post_mean[1] < 1.0
        assert 0.02 < res.post_mean[2] < 0.6

    def test_deterministic(self):
        r, _ = simulate_sv(200, seed=10)
        a = pmmh_sv(r, n_iter=50, n_particles=60, seed=11)
        b = pmmh_sv(r, n_iter=50, n_particles=60, seed=11)
        np.testing.assert_allclose(a.post_mean, b.post_mean)
        assert a.acceptance_rate == b.acceptance_rate

    def test_fail_closed(self):
        with pytest.raises(ValueError):
            pmmh_sv(np.arange(10.0), n_iter=50)
        with pytest.raises(ValueError):
            pmmh_sv(np.random.default_rng(0).standard_normal(50), n_iter=10)
        with pytest.raises(ValueError):
            pmmh_sv(
                np.random.default_rng(0).standard_normal(50),
                n_iter=50,
                theta0=np.array([-4.0, 1.5, 0.2]),
            )


class TestBench:
    def test_keys_finite(self):
        out = bench_pmcmc_sv(20260131)
        for k, v in out.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float)
            assert np.isfinite(v), k

    def test_deterministic(self):
        assert bench_pmcmc_sv(20260131) == bench_pmcmc_sv(20260131)

    def test_quality(self):
        out = bench_pmcmc_sv(20260131)
        assert out["synthetic_acceptance_rate"] > 0.02
        assert out["synthetic_determinism"] == 1.0
