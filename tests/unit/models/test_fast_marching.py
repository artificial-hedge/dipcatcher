import numpy as np

from quant_fund.models.fast_marching import (
    bench_fast_marching,
    fast_marching,
)


def test_fmm_radial():
    n = 41
    c = n // 2
    T = fast_marching(n, n, [(c, c, 0.0)])
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    d = np.sqrt((ii - c) ** 2 + (jj - c) ** 2)
    # axial direction is exact in the limit
    assert abs(T[c, 0] - c) < 0.5
    rel = np.abs(T[d > 2] - d[d > 2]) / d[d > 2]
    assert rel.mean() < 0.05


def test_fmm_two_sources():
    n = 41
    T = fast_marching(n, n, [(10, 10, 0.0), (30, 30, 0.0)])
    assert abs(T[10, 10]) < 1e-9
    assert T[20, 20] < T[5, 35]  # equidistant-ish cells ordered


def test_fmm_blocked_cells():
    n = 21
    blocked = np.zeros((n, n), dtype=np.bool_)
    blocked[:, 10] = True
    T = fast_marching(n, n, [(5, 5, 0.0)], blocked=blocked)
    assert np.isinf(T[15, 15])  # sealed off


def test_bench():
    out = bench_fast_marching(seed=5)
    assert out["synthetic_fmm_mean_rel_err"] < 0.08
    assert out["synthetic_fmm_detour"] > 1.5
    assert out["synthetic_fmm_wall_finite"] == 1.0
