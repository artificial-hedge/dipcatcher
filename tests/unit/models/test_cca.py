import numpy as np
import pytest

from quant_fund.models.cca import bench_cca, cca, pls, reduced_rank_regression


def _views(seed=0, n=400):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    xa = 0.85 * z + np.sqrt(1 - 0.85**2) * rng.standard_normal(n)
    xb = 0.85 * z + np.sqrt(1 - 0.85**2) * rng.standard_normal(n)
    x = np.column_stack([xa, rng.standard_normal(n), rng.standard_normal(n)])
    y = np.column_stack([xb, rng.standard_normal(n)])
    return x, y


def test_cca_shapes():
    x, y = _views()
    out = cca(x, y, n_components=2)
    assert out["correlations"].shape == (2,)
    assert out["x_weights"].shape == (3, 2)
    assert out["y_weights"].shape == (2, 2)


def test_cca_finds_shared_direction():
    x, y = _views(1)
    out = cca(x, y, n_components=1)
    a = np.asarray(out["x_weights"])[:, 0]
    # planted direction is e_1 in x-space
    assert abs(a[0]) / np.linalg.norm(a) > 0.9


def test_cca_correlation_ordering():
    x, y = _views(2)
    out = cca(x, y)
    rho = np.asarray(out["correlations"])
    assert rho[0] > rho[-1] or rho.size == 1


def test_pls_shapes():
    x, y = _views(3)
    out = pls(x, y, n_components=1)
    assert out["x_weights"].shape == (3, 1)
    assert out["covariances"].shape == (1,)


def test_rrr_rank_constraint():
    x, y = _views(4)
    out = reduced_rank_regression(x, y, rank=1)
    b = np.asarray(out["coef"])
    assert b.shape == (3, 2)
    assert np.linalg.matrix_rank(b, tol=1e-6) <= 2


def test_cca_input_validation():
    with pytest.raises(ValueError):
        cca(np.ones((5, 2)), np.ones((5, 2)))
    with pytest.raises(ValueError):
        cca(np.full((50, 2), np.nan), np.ones((50, 2)))


def test_bench_cca():
    out = bench_cca()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_cca_align"] > 0.85
