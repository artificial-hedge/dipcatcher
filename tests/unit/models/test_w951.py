"""Wave-951 matrix-function canon tests."""

from __future__ import annotations

from quant_fund.models.determinant_cofactor import bench_determinant_cofactor
from quant_fund.models.frechet_derivative import bench_frechet_derivative
from quant_fund.models.kronecker_sum import bench_kronecker_sum
from quant_fund.models.matrix_exponential import bench_matrix_exponential
from quant_fund.models.permanent_matrix import bench_permanent_matrix
from quant_fund.models.vec_operator import bench_vec_operator


def test_determinant_cofactor():
    assert bench_determinant_cofactor()["synthetic_determinant_cofactor"] == 1.0


def test_permanent_matrix():
    assert bench_permanent_matrix()["synthetic_permanent_matrix"] == 1.0


def test_matrix_exponential():
    assert bench_matrix_exponential()["synthetic_matrix_exponential"] == 1.0


def test_frechet_derivative():
    assert bench_frechet_derivative()["synthetic_frechet_derivative"] == 1.0


def test_vec_operator():
    assert bench_vec_operator()["synthetic_vec_operator"] == 1.0


def test_kronecker_sum():
    assert bench_kronecker_sum()["synthetic_kronecker_sum"] == 1.0
