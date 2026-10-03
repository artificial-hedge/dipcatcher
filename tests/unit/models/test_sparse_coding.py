import numpy as np

from quant_fund.models.sparse_coding import (
    bench_sparse_coding,
    ksvd,
    omp,
    omp_solve,
)


def test_omp_exact_support():
    d = np.eye(12)
    y = d[:, 3] * 2 + d[:, 7]
    idx = omp(d, y, 3)
    assert set(idx.tolist()) == {3, 7}


def test_omp_solve_coefficients():
    rng = np.random.default_rng(1)
    d = rng.normal(0, 1, (16, 8))
    d /= np.linalg.norm(d, axis=0, keepdims=True)
    c = np.zeros(8)
    c[[1, 5]] = [1.5, -2.0]
    y = d @ c
    est = omp_solve(d, y, 4)
    assert np.linalg.norm(y - d @ est) < 1e-6


def test_ksvd_reconstructs():
    rng = np.random.default_rng(2)
    d = np.eye(8)
    x = d[:, :4] @ rng.normal(0, 1, (4, 40))
    m = ksvd(x, n_atoms=6, sparsity=2, it=10, seed=2)
    recon = np.asarray(m["dictionary"]) @ np.asarray(m["codes"])
    assert np.linalg.norm(x - recon) / np.linalg.norm(x) < 0.3


def test_bench_sparse_coding():
    out = bench_sparse_coding(seed=562)
    assert out["synthetic_omp_support_hit"] >= 80
    assert out["synthetic_ksvd_relerr"] < 0.2
