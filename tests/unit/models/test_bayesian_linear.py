import numpy as np

from quant_fund.models.bayesian_linear import (
    ard_rvm_fit,
    bayes_lm_fit,
    bayes_lm_predict,
    bench_bayesian_linear,
)


def test_bayes_lm_recovers_coef():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (150, 3))
    y = 2 * x[:, 0] - x[:, 1] + 0.5 + rng.normal(0, 0.1, 150)
    m = bayes_lm_fit(x, y, it=200)
    mu, var = bayes_lm_predict(m, x)
    assert mu.shape == (150,) and var.shape == (150,)
    assert np.sqrt(((y - mu) ** 2).mean()) < 0.3


def test_ard_prunes_noise():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (120, 8))
    y = x[:, 0] * 2 + rng.normal(0, 0.2, 120)
    rvm = ard_rvm_fit(x, y, it=300)
    kept = np.asarray(rvm["kept"])
    assert kept[0]
    assert kept[1:6].sum() <= 2


def test_bench_bayesian_linear():
    out = bench_bayesian_linear(seed=558)
    assert out["synthetic_blm_rmse"] < 0.6
    assert out["synthetic_ard_noise_kept"] <= 3
