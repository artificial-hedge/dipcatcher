"""Tests for Hamiltonian Monte Carlo + NUTS (models/hmc.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.hmc import (
    DualAverage,
    bench_hmc,
    ess,
    hmc_sample,
    leapfrog,
    nuts_sample,
    rhat,
    synth_funnel,
    synth_gaussian,
    synth_student_t,
)


@pytest.fixture
def gauss():
    return synth_gaussian(dim=3, corr=0.8, seed=1)


def test_leapfrog_reversibility(gauss):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(3)
    p = rng.standard_normal(3)
    x1, p1 = leapfrog(gauss, x, p, 0.1, 20)
    x2, p2 = leapfrog(gauss, x1, -p1, 0.1, 20)
    assert np.allclose(x2, x, atol=1e-8)
    assert np.allclose(p2, -p, atol=1e-8)


def test_leapfrog_energy_bound(gauss):
    rng = np.random.default_rng(0)
    x = np.zeros(3)
    p = rng.standard_normal(3)
    lp0, _ = gauss(x)
    h0 = -lp0 - 0.5 * p @ p
    x1, p1 = leapfrog(gauss, x, p, 0.05, 50)
    lp1, _ = gauss(x1)
    h1 = -lp1 - 0.5 * p1 @ p1
    assert abs(h1 - h0) < 0.5


def test_leapfrog_diverges_flagged(gauss):
    with pytest.raises(FloatingPointError):
        leapfrog(gauss, np.zeros(3), np.ones(3) * 1e200, 1.0, 10)


def test_hmc_gaussian_moments(gauss):
    d, acc = hmc_sample(gauss, np.zeros(3), eps=0.15, n_leap=15, n_draws=400, burn=100, seed=3)
    assert d.shape == (400, 3)
    assert acc.mean() > 0.8
    assert np.abs(d.mean(axis=0)).max() < 0.4
    assert np.abs(d.std(axis=0) - 1.0).max() < 0.4


def test_hmc_accept_affected_by_eps(gauss):
    _d1, acc_small = hmc_sample(
        gauss, np.zeros(3), eps=0.05, n_leap=10, n_draws=100, burn=50, seed=3
    )
    _d2, acc_big = hmc_sample(gauss, np.zeros(3), eps=1.2, n_leap=10, n_draws=100, burn=50, seed=3)
    assert acc_small.mean() > acc_big.mean()


def test_nuts_moves(gauss):
    d, depths, eerrs = nuts_sample(
        gauss, np.zeros(3), n_draws=300, burn=100, eps=0.3, adapt=False, seed=7
    )
    assert d.shape == (300, 3)
    assert np.abs(d).max() > 0.5
    assert depths.mean() > 1.0
    assert np.std(d[:, 0]) > 0.3


def test_nuts_gaussian_moments(gauss):
    d, _dep, _ee = nuts_sample(
        gauss, np.zeros(3), n_draws=500, burn=200, eps=0.3, adapt=False, seed=8
    )
    assert np.abs(d.mean(axis=0)).max() < 0.3
    assert np.abs(d.std(axis=0) - 1.0).max() < 0.35


def test_nuts_determinism(gauss):
    a = nuts_sample(gauss, np.zeros(3), n_draws=60, burn=20, eps=0.3, seed=11)[0]
    b = nuts_sample(gauss, np.zeros(3), n_draws=60, burn=20, eps=0.3, seed=11)[0]
    assert np.array_equal(a, b)


def test_nuts_adapted_smoke(gauss):
    d, _dep, _ee = nuts_sample(
        gauss, np.zeros(3), n_draws=120, burn=80, eps=0.3, adapt=True, seed=13
    )
    assert np.all(np.isfinite(d))
    assert np.abs(d.mean(axis=0)).max() < 0.8


def test_funnel_reaches_neck():
    f = synth_funnel(dim=3)
    d, _dep, _ee = nuts_sample(
        f,
        np.zeros(3),
        n_draws=800,
        burn=200,
        eps=0.2,
        max_depth=7,
        target_accept=0.85,
        eps_max=0.8,
        seed=17,
    )
    # y ~ N(0,9): the neck region y < -3 must be visited
    assert np.mean(d[:, 0] < -3.0) > 0.05


def test_dual_average_moves():
    da = DualAverage(0.2, target_accept=0.8)
    eps_up = da.step(0.95, 1)
    da2 = DualAverage(0.2, target_accept=0.8)
    eps_dn = da2.step(0.2, 1)
    assert eps_up != eps_dn
    assert da.final_eps > 0


def test_ess_uncorrelated():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    assert ess(x) > 800


def test_ess_correlated_lower():
    rng = np.random.default_rng(0)
    e = rng.standard_normal(2000)
    x = np.empty(2000)
    x[0] = 0.0
    for i in range(1, 2000):
        x[i] = 0.95 * x[i - 1] + e[i]
    assert ess(x) < ess(e) * 0.5


def test_rhat_identical_chains():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(300)
    chains = np.stack([base, base + 0.001, base - 0.001, base * 1.001])
    assert 0.9 < rhat(chains) < 1.1


def test_rhat_detects_drift():
    rng = np.random.default_rng(0)
    chains = np.stack([rng.standard_normal(300) + s * 5.0 for s in range(4)])
    assert rhat(chains) > 2.0


def test_student_t_target_samples():
    f = synth_student_t(df=4.0, dim=2)
    d, _acc = hmc_sample(f, np.zeros(2), eps=0.1, n_leap=12, n_draws=300, burn=100, seed=5)
    assert np.abs(d.mean(axis=0)).max() < 0.4


def test_bench_keys():
    out = bench_hmc()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_hmc_ess_per_draw"] > 0.5
    assert out["synthetic_nuts_ess_per_draw"] > 0.05
    assert out["synthetic_funnel_capture"] > 0.05
    assert out["synthetic_rhat"] < 1.5
    assert out["synthetic_determinism"] == 1.0


def test_input_validation(gauss):
    with pytest.raises(ValueError):
        hmc_sample(gauss, np.zeros(3), eps=-1.0)
    with pytest.raises(ValueError):
        nuts_sample(gauss, np.zeros(3), n_draws=0)
    with pytest.raises(ValueError):
        leapfrog(gauss, np.array([]), np.array([]), 0.1, 1)
    with pytest.raises(ValueError):
        rhat(np.zeros((1, 10)))
