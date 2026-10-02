import numpy as np

from quant_fund.models.gp_classification import (
    bench_gp_classification,
    gpc_fit,
    gpc_predict,
)


def test_gpc_separable():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal([0, 0], 0.3, (40, 2)), rng.normal([3, 3], 0.3, (40, 2))])
    y = np.r_[np.zeros(40), np.ones(40)]
    m = gpc_fit(x, y, ls=0.8, sf2=2.0)
    p = gpc_predict(m, x)
    assert ((p > 0.5).astype(float) == y).mean() > 0.95


def test_gpc_prob_range():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (60, 2))
    y = (x[:, 0] > 0).astype(float)
    m = gpc_fit(x, y)
    p = gpc_predict(m, x)
    assert ((p >= 0) & (p <= 1)).all()


def test_bench_gp_classification():
    out = bench_gp_classification(seed=549)
    assert out["synthetic_gpc_auc"] > 0.9
    assert out["synthetic_gpc_auc"] > out["synthetic_logit_auc"]
