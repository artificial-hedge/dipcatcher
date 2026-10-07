"""Probes for _pd_synth (banana posterior fixture)."""

import numpy as np
import pytest

from quant_fund.models._pd_synth import (
    DIM,
    ess_1d,
    grad_logp,
    logp,
    mean_ess,
    moment_err,
    ref_moments,
    rwm_baseline,
)


def test_logp_mode_and_gradient_consistency():
    # mode of banana: x0=0 forces x_i=1 for i>=1
    x_mode = np.array([0.0, 1.0, 1.0, 1.0])
    x_off = np.array([0.1, 1.0, 1.0, 1.0])
    assert logp(x_mode) > logp(x_off)
    # finite-difference check of grad_logp
    x = np.array([0.3, 0.8, 1.2, 0.9])
    g = grad_logp(x)
    eps = 1e-6
    fd = np.array([(logp(x + eps * e) - logp(x - eps * e)) / (2 * eps) for e in np.eye(4)])
    np.testing.assert_allclose(g, fd, atol=1e-4)


@pytest.mark.parametrize("x", [np.zeros(0), np.zeros(1), np.array([np.nan, 1.0])])
def test_logp_hostile(x):
    with pytest.raises(ValueError):
        logp(x)


@pytest.mark.parametrize("x", [np.zeros(1), np.array([np.inf, 1.0])])
def test_grad_logp_hostile(x):
    with pytest.raises(ValueError):
        grad_logp(x)


def test_ref_moments_deterministic_small():
    mu, sd = ref_moments(seed=5, n=2000)
    assert mu.shape == (DIM,) and sd.shape == (DIM,)
    assert np.isfinite(mu).all() and (sd > 0).all()
    mu2, _ = ref_moments(seed=5, n=2000)
    np.testing.assert_array_equal(mu, mu2)


def test_ref_moments_rejects_tiny_n():
    with pytest.raises(ValueError):
        ref_moments(n=4)


def test_rwm_baseline_shape_and_determinism():
    a = rwm_baseline(0, n=500)
    b = rwm_baseline(0, n=500)
    assert a.shape == (500, DIM)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "kw", [{"n": 0}, {"n": -2}, {"step": 0.0}, {"step": -0.1}, {"step": np.nan}]
)
def test_rwm_baseline_hostile(kw):
    base = {"n": 10}
    base.update(kw)
    with pytest.raises(ValueError):
        rwm_baseline(0, **base)


def test_moment_err_zero_at_truth():
    s = rwm_baseline(0, n=100)
    mu = s[25:].mean(0)
    sd = s[25:].std(0)
    assert moment_err(s, mu, sd) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "s,mu,sd",
    [
        (np.zeros((3, 4)), np.zeros(4), np.ones(4)),  # too few rows
        (np.zeros((10, 4)), np.zeros(3), np.ones(3)),  # dim mismatch
        (np.zeros(10), np.zeros(1), np.ones(1)),  # 1-D samples
    ],
)
def test_moment_err_hostile(s, mu, sd):
    with pytest.raises(ValueError):
        moment_err(s, mu, sd)


def test_ess_1d_iid_vs_correlated():
    rng = np.random.default_rng(0)
    iid = rng.standard_normal(2000)
    ar = np.zeros(2000)
    for i in range(1, 2000):
        ar[i] = 0.98 * ar[i - 1] + rng.standard_normal()
    assert ess_1d(iid) > ess_1d(ar)


def test_mean_ess_finite():
    s = rwm_baseline(0, n=200)
    e = mean_ess(s)
    assert 0 < e <= 200


def test_mean_ess_rejects_empty():
    with pytest.raises(ValueError):
        mean_ess(np.zeros((0, 4)))
