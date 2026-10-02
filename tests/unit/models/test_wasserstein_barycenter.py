import numpy as np

from quant_fund.models.wasserstein_barycenter import (
    barycenter_fixed_support,
    bench_wasserstein_barycenter,
)


def test_identical_measures_recover():
    n = 30
    grid = np.linspace(0.0, 1.0, n)
    cost = (grid[:, None] - grid[None, :]) ** 2
    d = np.exp(-0.5 * ((grid - 0.4) / 0.05) ** 2)
    m = d / d.sum()
    bar = barycenter_fixed_support([m, m, m], cost, eps=0.005, max_iter=800)
    assert 0.5 * np.abs(bar - m).sum() < 0.2
    assert abs(grid[int(np.argmax(bar))] - 0.4) < 0.05


def test_bench_peak_between_clusters():
    out = bench_wasserstein_barycenter(seed=6)
    assert 0.3 < out["synthetic_bary_peak"] < 0.6
    assert abs(out["synthetic_bary_mass"] - 1.0) < 1e-9
