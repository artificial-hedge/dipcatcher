import numpy as np

from quant_fund.models.collaborative_filtering import (
    als_predict,
    als_wr_fit,
    bench_collaborative_filtering,
    bias_mf_fit,
    bias_mf_predict,
    item_knn_predict,
)


def _ratings(seed: int = 0):
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 1, (30, 2))
    q = rng.normal(0, 1, (20, 2))
    full = p @ q.T + 3
    mask = rng.uniform(0, 1, (30, 20)) < 0.6
    return np.where(mask, full, 0.0)


def test_als_fits_observed():
    r = _ratings()
    m = als_wr_fit(r, n_factors=3, it=12, seed=0)
    pred = als_predict(m)
    obs = r != 0
    assert np.sqrt(((r[obs] - pred[obs]) ** 2).mean()) < 1.5


def test_bias_mf_fits():
    r = _ratings(1)
    m = bias_mf_fit(r, n_factors=3, it=25, seed=1)
    pred = bias_mf_predict(m)
    obs = r != 0
    assert np.sqrt(((r[obs] - pred[obs]) ** 2).mean()) < 1.5


def test_knn_shape():
    r = _ratings(2)
    pred = item_knn_predict(r, k=5)
    assert pred.shape == r.shape


def test_bench_collaborative_filtering():
    out = bench_collaborative_filtering(seed=567)
    assert out["synthetic_cf_als_rmse"] < out["synthetic_cf_mean_rmse"]
