"""Wave-949 matrix-pencil canon tests."""

from __future__ import annotations

from quant_fund.models.deflating_subspace import bench_deflating_subspace
from quant_fund.models.invariant_subspace import bench_invariant_subspace
from quant_fund.models.jordan_form import bench_jordan_form
from quant_fund.models.kronecker_canonical import bench_kronecker_canonical
from quant_fund.models.matrix_pencil import bench_matrix_pencil
from quant_fund.models.rational_canonical import bench_rational_canonical


def test_matrix_pencil():
    assert bench_matrix_pencil()["synthetic_matrix_pencil"] == 1.0


def test_kronecker_canonical():
    assert bench_kronecker_canonical()["synthetic_kronecker_canonical"] == 1.0


def test_invariant_subspace():
    assert bench_invariant_subspace()["synthetic_invariant_subspace"] == 1.0


def test_deflating_subspace():
    assert bench_deflating_subspace()["synthetic_deflating_subspace"] == 1.0


def test_jordan_form():
    assert bench_jordan_form()["synthetic_jordan_form"] == 1.0


def test_rational_canonical():
    assert bench_rational_canonical()["synthetic_rational_canonical"] == 1.0
