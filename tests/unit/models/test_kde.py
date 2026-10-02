import numpy as np

from quant_fund.models.kde import (
    bench_kde,
    kde_cdf,
    kde_eval,
    loo_cv_bw,
)


def test_kde_eval_matches_density():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.5, 300)
    g = np.linspace(-2, 2, 50)
    f = kde_eval(x, g, 0.4)
    assert f.shape == g.shape
    assert np.all(f >= 0)
    dx = g[1] - g[0]
    assert abs(f.sum() * dx - 1.0) < 0.05


def test_kde_cdf_monotone():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    g = np.linspace(-3, 3, 60)
    c = kde_cdf(x, g, 0.5)
    assert np.all(np.diff(c) >= -1e-10)
    assert c[-1] > 0.95


def test_loo_bw_reasonable():
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, 300)
    b, _ = loo_cv_bw(x)
    assert 0.1 < b < 2.0


def test_bench_keys():
    out = bench_kde()
    assert out["synthetic_kde_l2_cv"] < 0.15
    assert out["synthetic_kde_l2_cv"] <= out["synthetic_kde_l2_silv"] + 0.05
    assert out["synthetic_cdf_tail_err"] < 1e-6
