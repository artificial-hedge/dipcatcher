import numpy as np

from quant_fund.models.gibbs_sampler import (
    bench_gibbs_sampler,
    gibbs_lm,
    gibbs_normal_mean,
)


def test_gibbs_mean_recovers():
    rng = np.random.default_rng(0)
    y = rng.normal(2.0, 0.5, 100)
    g = gibbs_normal_mean(y, it=2000, burn=300, seed=0)
    assert abs(g["mu"].mean() - 2.0) < 0.15
    assert g["tau"].mean() > 0


def test_gibbs_lm_recovers():
    rng = np.random.default_rng(1)
    x = np.column_stack([np.ones(150), rng.standard_normal(150)])
    y = x @ np.array([1.0, -1.5]) + 0.3 * rng.standard_normal(150)
    g = gibbs_lm(x, y, it=2000, burn=300, seed=1)
    assert np.linalg.norm(g["beta"].mean(0) - [1.0, -1.5]) < 0.2


def test_bench_keys():
    out = bench_gibbs_sampler()
    assert out["synthetic_gibbs_mu_err"] < 0.2
    assert out["synthetic_gibbs_beta_err"] < 0.2
    assert out["synthetic_gibbs_sigma_err"] < 0.15
