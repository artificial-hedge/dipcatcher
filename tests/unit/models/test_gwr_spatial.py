"""GWR tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.gwr_spatial import (
    bench_gwr,
    bisquare_kernel,
    gaussian_kernel,
    gwr_cv,
    gwr_fit,
)


def _surface(seed=0, n=100):
    rng = np.random.default_rng(seed)
    c = rng.uniform(0, 1, (n, 2))
    X = rng.normal(0, 1, (n, 1))
    beta = 1 + 2 * c[:, 0]
    y = 1.0 + X[:, 0] * beta + rng.normal(0, 0.4, n)
    return c, X, y, beta


def test_gwr_local_betas_vary():
    c, X, y, beta = _surface()
    fit = gwr_fit(c, X, y, knn=40)
    assert fit["betas"].shape == (100, 1)
    # spatially varying truth -> spread in local betas
    # (GWR smooths, so observed spread < truth's 0.58 sd)
    assert fit["betas"][:, 0].std() > 0.1


def test_gwr_stationary_recovers_constant():
    rng = np.random.default_rng(2)
    c = rng.uniform(0, 1, (80, 2))
    X = rng.normal(0, 1, (80, 1))
    y = 1.0 + 2.0 * X[:, 0] + rng.normal(0, 0.3, 80)
    fit = gwr_fit(c, X, y, knn=60)
    assert np.abs(fit["betas"][:, 0] - 2.0).mean() < 0.3


def test_kernels():
    d = np.array([0.0, 0.5, 1.0, 2.0])
    g = gaussian_kernel(d, 1.0)
    assert g[0] == 1.0 and g[-1] < 0.2
    b = bisquare_kernel(d, 1.0)
    assert b[0] == 1.0 and b[2] == 0.0 and b[3] == 0.0


def test_cv_picks_bandwidth():
    c, X, y, _ = _surface()
    out = gwr_cv(c, X, y, knn_grid=[30, 60, 90])
    assert out["knn"] in (30, 60, 90)


def test_bench_gwr():
    out = bench_gwr()
    assert out["synthetic_corr_b1"] > 0.8
