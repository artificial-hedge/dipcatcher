"""Wave-255 optimization-2 canon tests."""

import numpy as np

from quant_fund.models.admm_lasso import _soft, bench_admm_lasso
from quant_fund.models.barrier_ip import barrier_lp, bench_barrier_ip
from quant_fund.models.coord_descent import bench_coord_descent, coord_descent_qp
from quant_fund.models.ellipsoid_method import bench_ellipsoid_method, ellipsoid_feasible
from quant_fund.models.proj_gradient import _proj_simplex, bench_proj_gradient
from quant_fund.models.simplex_lp import bench_simplex_lp, simplex


def test_simplex_basic():
    A = np.array([[1.0, 1.0]])
    b = np.array([4.0])
    c = np.array([3.0, 2.0])
    obj, x = simplex(c, A, b)
    assert np.isclose(obj, 12.0)
    assert np.allclose(x, [4.0, 0.0])


def test_simplex_bench():
    assert bench_simplex_lp()["synthetic_optimum_exact"] > 0.9


def test_ellipsoid_basic():
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([1.0, 1.0])
    x = ellipsoid_feasible(A, b)
    assert x is not None


def test_ellipsoid_bench():
    assert bench_ellipsoid_method()["synthetic_feasible_point"] == 1.0


def test_barrier_basic():
    A = np.array([[1.0, 1.0]])
    b = np.array([2.0])
    c = np.array([1.0, 3.0])
    x = barrier_lp(c, A, b)
    assert np.allclose(A @ x, b, atol=1e-3)


def test_barrier_bench():
    assert bench_barrier_ip()["synthetic_ip_optimal"] > 0.9


def test_admm_soft():
    assert np.allclose(_soft(np.array([2.0, -0.5]), 1.0), [1.0, 0.0])


def test_admm_bench():
    assert bench_admm_lasso()["synthetic_admm_optimal"] == 1.0


def test_coord_basic():
    Q = np.eye(2)
    c = np.array([1.0, 2.0])
    x = coord_descent_qp(Q, c)
    assert np.allclose(x, -c)


def test_coord_bench():
    assert bench_coord_descent()["synthetic_qp_exact"] == 1.0


def test_proj_simplex():
    v = _proj_simplex(np.array([0.5, -1.0, 2.0]))
    assert np.isclose(v.sum(), 1.0)


def test_proj_bench():
    assert bench_proj_gradient()["synthetic_simplex_optimal"] == 1.0
