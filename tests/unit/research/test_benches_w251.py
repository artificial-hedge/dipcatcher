"""Wave-251 adapter tests."""

from quant_fund.research.benches_w251 import (
    bench_givens_qr_family,
    bench_jacobi_svd_family,
    bench_ldlt_solve_family,
    bench_lu_pivots_family,
    bench_orth_iter_family,
    bench_sturm_eig_family,
)


def test_bench_givens_qr_family():
    out = bench_givens_qr_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_jacobi_svd_family():
    out = bench_jacobi_svd_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_ldlt_solve_family():
    out = bench_ldlt_solve_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_lu_pivots_family():
    out = bench_lu_pivots_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_orth_iter_family():
    out = bench_orth_iter_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_sturm_eig_family():
    out = bench_sturm_eig_family()
    assert out and all(k.startswith("synthetic_") for k in out)
