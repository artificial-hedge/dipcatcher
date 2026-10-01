import numpy as np
import pytest

from quant_fund.models.isomap import bench_isomap, classical_mds, isomap


def _roll(seed=0, n=120):
    rng = np.random.default_rng(seed)
    s_lat = rng.uniform(0.5, 4.5, n)
    t_lat = rng.uniform(0.0, 1.0, n)
    x = np.column_stack([s_lat * np.cos(s_lat), 10.0 * t_lat, s_lat * np.sin(s_lat)])
    return x, s_lat


def test_isomap_shapes():
    x, _ = _roll()
    out = isomap(x, k=10, n_components=2)
    assert out["coords"].shape == (120, 2)
    assert out["geodesics"].shape == (120, 120)


def test_isomap_geodesics_exceed_euclidean():
    x, _ = _roll(1)
    out = isomap(x, k=10, n_components=2)
    d_geo = np.asarray(out["geodesics"])
    # geodesic distance must dominate straight-line distance
    d_e = np.sqrt(np.sum((x[:, None] - x[None]) ** 2, axis=2))
    assert (d_geo >= d_e - 1e-9).all()


def test_isomap_recovers_ordering():
    from scipy.stats import spearmanr

    x, s_lat = _roll(2, n=200)
    out = isomap(x, k=12, n_components=2)
    coords = np.asarray(out["coords"])
    rho = max(abs(float(spearmanr(coords[:, j], s_lat)[0])) for j in range(2))
    assert rho > 0.7


def test_classical_mds_line():
    # points on a line: 2-D MDS of geodesics should recover rank ~1
    t = np.linspace(0, 1, 30)
    d2 = (t[:, None] - t[None, :]) ** 2
    out = classical_mds(d2, n_components=2)
    w = np.asarray(out["eigenvalues"])
    assert w[0] > 10 * abs(w[1])


def test_isomap_input_validation():
    with pytest.raises(ValueError):
        isomap(np.ones((4, 3)), k=10)
    with pytest.raises(ValueError):
        isomap(np.full((50, 3), np.nan), k=5)


def test_bench_isomap():
    out = bench_isomap()
    assert out["score"] == 1.0
    assert out["synthetic_isomap_order_rho"] > 0.8
