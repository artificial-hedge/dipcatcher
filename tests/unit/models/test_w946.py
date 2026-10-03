"""Wave-946 positive-matrix canon tests."""

from __future__ import annotations

from quant_fund.models.cholesky_piv import bench_cholesky_piv
from quant_fund.models.douglas_factor import bench_douglas_factor
from quant_fund.models.matrix_square_root import bench_matrix_square_root
from quant_fund.models.perron_frobenius import bench_perron_frobenius
from quant_fund.models.polar_decomp import bench_polar_decomp
from quant_fund.models.sylvester_matrix import bench_sylvester_matrix


def test_perron_frobenius():
    assert bench_perron_frobenius()["synthetic_perron_frobenius"] == 1.0


def test_douglas_factor():
    assert bench_douglas_factor()["synthetic_douglas_factor"] == 1.0


def test_cholesky_piv():
    assert bench_cholesky_piv()["synthetic_cholesky_piv"] == 1.0


def test_matrix_square_root():
    assert bench_matrix_square_root()["synthetic_matrix_square_root"] == 1.0


def test_polar_decomp():
    assert bench_polar_decomp()["synthetic_polar_decomp"] == 1.0


def test_sylvester_matrix():
    assert bench_sylvester_matrix()["synthetic_sylvester_matrix"] == 1.0
