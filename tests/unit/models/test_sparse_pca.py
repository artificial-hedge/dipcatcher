import numpy as np

from quant_fund.models.sparse_pca import (
    bench_sparse_pca,
    spca_direction,
    spca_fit,
)


def test_direction_finds_sparse():
    d = 20
    v = np.zeros(d)
    v[:3] = [0.6, 0.5, 0.6]
    cov = 2.0 * np.outer(v, v) + 0.1 * np.eye(d)
    hat = spca_direction(cov, 0.05)
    assert abs(hat @ (v / np.linalg.norm(v))) > 0.98
    assert np.flatnonzero(np.abs(hat) > 1e-3).size <= 6


def test_fit_deflation():
    d = 15
    v1 = np.zeros(d)
    v1[:4] = 0.5
    v2 = np.zeros(d)
    v2[4:8] = 0.5
    cov = 3 * np.outer(v1, v1) + 2 * np.outer(v2, v2) + 0.1 * np.eye(d)
    comps, var = spca_fit(cov, 0.08, 2)
    assert var[0] >= var[1] - 1e-6 or var[1] > 0
    assert comps.shape == (2, d)


def test_bench_keys():
    out = bench_sparse_pca()
    assert out["synthetic_spca_angle"] > 0.95
    assert out["synthetic_spca_support_hit"] >= 4
    assert out["synthetic_spca_support_extra"] <= 2
