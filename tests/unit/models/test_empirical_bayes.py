"""Empirical Bayes: Robbins, Tweedie, KW-NPMLE."""

from __future__ import annotations

import numpy as np

from quant_fund.models.empirical_bayes import (
    bench_eb,
    kw_npmle,
    robbins_poisson,
    tweedie_mean,
)


def test_robbins_beats_naive():
    rng = np.random.default_rng(538)
    lam = np.repeat([2.0, 8.0], 250)
    x = rng.poisson(lam)
    eb = robbins_poisson(x)
    assert np.mean((eb - lam) ** 2) < np.mean((x - lam) ** 2)


def test_robbins_shrinks_toward_prior():
    # sparse high-count cell: smoothed ratio pulls
    # extreme observations back toward the mass
    rng = np.random.default_rng(538)
    lam = np.repeat([2.0, 8.0], 250)
    x = rng.poisson(lam)
    eb = robbins_poisson(x)
    hi = x >= 10
    assert eb[hi].mean() < x[hi].mean()


def test_kw_npmle_two_point_mixture():
    rng = np.random.default_rng(4)
    mus = np.repeat([-2.0, 3.0], 150)
    x = mus + rng.normal(size=300)
    r = kw_npmle(x, sigma=1.0, it=300)
    w = np.asarray(r["weights"])
    g = np.asarray(r["grid"])
    top = g[np.argsort(w)[-2:]]
    dist = min(abs(top[0] + 2), abs(top[0] - 3)) + min(abs(top[1] + 2), abs(top[1] - 3))
    assert dist < 1.0


def test_kw_weights_sum_to_one():
    rng = np.random.default_rng(8)
    x = rng.normal(size=100)
    r = kw_npmle(x, sigma=1.0, it=50)
    assert abs(float(np.asarray(r["weights"]).sum()) - 1.0) < 1e-8


def test_tweedie_shrinks_extremes():
    rng = np.random.default_rng(6)
    z = np.concatenate([rng.normal(size=300), np.array([4.0, -4.0])])
    pm = np.asarray(tweedie_mean(z, sigma=1.0)["post_mean"])
    assert abs(pm[-2]) < 4.0
    assert abs(pm[-1]) < 4.0


def test_bench_eb():
    out = bench_eb(seed=538)
    assert out["synthetic_robbins_ratio"] < 1.0
    assert out["synthetic_kw_modes_dist"] < 1.0
