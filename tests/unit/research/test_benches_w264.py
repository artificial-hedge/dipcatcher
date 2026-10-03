"""Wave-264 adapter tests."""

from quant_fund.research.benches_w264 import (
    bench_block_lanczos_family,
    bench_divide_conquer_eig_family,
    bench_dqds_family,
    bench_fgmres_family,
    bench_randomized_qb_family,
    bench_sparse_cholesky_family,
)


def test_bench_divide_conquer_eig_family():
    assert all(k.startswith("synthetic_") for k in bench_divide_conquer_eig_family())


def test_bench_dqds_family():
    assert all(k.startswith("synthetic_") for k in bench_dqds_family())


def test_bench_block_lanczos_family():
    assert all(k.startswith("synthetic_") for k in bench_block_lanczos_family())


def test_bench_randomized_qb_family():
    assert all(k.startswith("synthetic_") for k in bench_randomized_qb_family())


def test_bench_sparse_cholesky_family():
    assert all(k.startswith("synthetic_") for k in bench_sparse_cholesky_family())


def test_bench_fgmres_family():
    assert all(k.startswith("synthetic_") for k in bench_fgmres_family())
