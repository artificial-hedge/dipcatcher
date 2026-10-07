import numpy as np
import pytest

from quant_fund.models.compositional import (
    bench_compositional,
    clr,
    dirichlet_moments,
    ilr,
    ilr_inv,
    variation_matrix,
)


def _dirichlet(seed=0, n=500):
    rng = np.random.default_rng(seed)
    return rng.dirichlet([2.0, 5.0, 1.5], size=n)


def test_clr_zero_sum():
    x = _dirichlet()
    c = clr(x)
    assert c.shape == x.shape
    assert np.abs(c.sum(axis=1)).max() < 1e-10


def test_ilr_shape_isometry():
    x = _dirichlet(1)
    z = ilr(x)
    assert z.shape == (x.shape[0], x.shape[1] - 1)
    # isometry: distances preserved between two rows
    import numpy.linalg as la

    c = clr(x)
    d_clr = la.norm(c[0] - c[1])
    d_ilr = la.norm(z[0] - z[1])
    assert abs(d_clr - d_ilr) < 1e-10


def test_ilr_inv_roundtrip():
    x = _dirichlet(2)
    back = ilr_inv(ilr(x))
    assert np.abs(back - x).max() < 1e-9


def test_variation_matrix_structure():
    x = _dirichlet(3)
    t = variation_matrix(x)
    assert t.shape == (3, 3)
    assert np.allclose(t, t.T)
    assert np.abs(np.diag(t)).max() < 1e-12
    assert (t >= -1e-12).all()


def test_dirichlet_moments_recovery():
    x = _dirichlet(4, n=800)
    out = dirichlet_moments(x)
    assert abs(out["alpha0"] - 8.5) / 8.5 < 0.25


def test_compositional_input_validation():
    with pytest.raises(ValueError):
        clr(np.array([[0.5, -0.1, 0.6]]))
    with pytest.raises(ValueError):
        clr(np.array([[0.5, 0.0, 0.5]]))
    with pytest.raises(ValueError):
        dirichlet_moments(np.array([[0.5, 0.5]]))


def test_bench_compositional():
    out = bench_compositional()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_coda_rec_err"] < 1e-9
