"""Wave-251 numerical-linalg-2 canon tests."""

import numpy as np

from quant_fund.models.givens_qr import bench_givens_qr, givens_qr
from quant_fund.models.jacobi_svd import bench_jacobi_svd, jacobi_svd
from quant_fund.models.ldlt_solve import bench_ldlt_solve, ldlt
from quant_fund.models.lu_pivots import bench_lu_pivots, lu_decompose
from quant_fund.models.orth_iter import bench_orth_iter, orth_iter
from quant_fund.models.sturm_eig import bench_sturm_eig, sturm_eigs


def test_ldlt_basic():
    A = np.array([[4.0, 2.0], [2.0, 3.0]])
    L, d = ldlt(A)
    assert np.allclose(L @ np.diag(d) @ L.T, A)


def test_ldlt_bench():
    assert bench_ldlt_solve()["synthetic_solve_exact"] == 1.0


def test_givens_basic():
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    Q, R = givens_qr(A)
    assert np.allclose(Q @ R, A)


def test_givens_bench():
    assert bench_givens_qr()["synthetic_factor_exact"] == 1.0


def test_jacobi_basic():
    A = np.eye(3)
    U, s, Vt = jacobi_svd(A)
    assert np.allclose(s, 1.0)


def test_jacobi_bench():
    assert bench_jacobi_svd()["synthetic_singular_values"] == 1.0


def test_orth_iter_basic():
    A = np.diag([3.0, 2.0, 1.0])
    ritz, Q = orth_iter(A, 1)
    assert np.isclose(ritz[0], 3.0)


def test_orth_iter_bench():
    assert bench_orth_iter()["synthetic_subspace_exact"] == 1.0


def test_lu_basic():
    A = np.array([[2.0, 1.0], [4.0, 3.0]])
    P, L, U = lu_decompose(A)
    assert np.allclose(P @ A, L @ U)


def test_lu_bench():
    assert bench_lu_pivots()["synthetic_factor_exact"] == 1.0


def test_sturm_basic():
    A = np.diag([1.0, 2.0, 3.0])
    eigs = sturm_eigs(A)
    assert np.allclose(np.sort(eigs), [1.0, 2.0, 3.0], atol=1e-4)


def test_sturm_bench():
    assert bench_sturm_eig()["synthetic_eigs_exact"] == 1.0
