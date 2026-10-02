import numpy as np

from quant_fund.models.nearest_centroid import (
    bench_nearest_centroid,
    ncm_fit,
    ncm_predict,
    nsc_active,
    nsc_fit,
    nsc_predict,
)


def _data(seed=0):
    rng = np.random.default_rng(seed)
    n, d = 240, 40
    y = np.tile(np.arange(3), n // 3).astype(np.int64)
    x = rng.standard_normal((n, d))
    for k in range(3):
        idx = np.flatnonzero(y == k)
        x[np.ix_(idx, np.arange(6))] += (k - 1) * 1.0
    return x, y


def test_ncm():
    x, y = _data()
    m = ncm_fit(x, y)
    assert np.mean(ncm_predict(m, x) == y) > 0.8


def test_nsc_sparsifies():
    x, y = _data()
    m = nsc_fit(x, y, 3.0)
    act = nsc_active(m)
    assert set(act.tolist()) == set(range(6))
    assert np.mean(nsc_predict(m, x) == y) > 0.8


def test_bench_keys():
    out = bench_nearest_centroid()
    assert out["synthetic_nsc_active_hit"] >= 7
    assert out["synthetic_nsc_active_extra"] <= 2
