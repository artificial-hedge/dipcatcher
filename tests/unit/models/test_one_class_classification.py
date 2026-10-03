import numpy as np

from quant_fund.models.one_class_classification import (
    bench_one_class,
    lof_score,
    mahalanobis_score,
    svdd_fit,
    svdd_score,
)


def test_svdd_flags_outlier():
    rng = np.random.default_rng(0)
    x_in = rng.normal(0, 0.5, (100, 2))
    m = svdd_fit(x_in, nu=0.1, seed=0)
    s_in = svdd_score(m, x_in)
    s_out = svdd_score(m, np.array([[6.0, 6.0]]))
    assert s_out[0] > np.median(s_in)


def test_mahalanobis_outlier_far():
    rng = np.random.default_rng(1)
    x_in = rng.normal(0, 1, (150, 3))
    s_in = mahalanobis_score(x_in, x_in)
    s_out = mahalanobis_score(x_in, np.array([[9.0, 9.0, 9.0]]))
    assert s_out[0] > np.percentile(s_in, 95)


def test_lof_flags_isolated():
    rng = np.random.default_rng(2)
    x = np.vstack([rng.normal(0, 0.3, (60, 2)), np.array([[4.0, 4.0]])])
    s = lof_score(x, k=8)
    assert s[-1] == s.max()


def test_bench_one_class():
    out = bench_one_class(seed=548)
    assert out["synthetic_svdd_auc"] > 0.85
    assert out["synthetic_lof_auc"] > 0.85
