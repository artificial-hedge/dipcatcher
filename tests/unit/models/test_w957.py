"""Wave-957 operator-theory-2 canon tests."""

from __future__ import annotations

from quant_fund.models.fredholm_op import bench_fredholm_op
from quant_fund.models.multiplication_op import bench_multiplication_op
from quant_fund.models.normal_operator import bench_normal_operator
from quant_fund.models.selfadjoint_op import bench_selfadjoint_op
from quant_fund.models.shift_operator import bench_shift_operator
from quant_fund.models.unitary_operator import bench_unitary_operator


def test_selfadjoint_op():
    assert bench_selfadjoint_op()["synthetic_selfadjoint_op"] == 1.0


def test_unitary_operator():
    assert bench_unitary_operator()["synthetic_unitary_operator"] == 1.0


def test_shift_operator():
    assert bench_shift_operator()["synthetic_shift_operator"] == 1.0


def test_fredholm_op():
    assert bench_fredholm_op()["synthetic_fredholm_op"] == 1.0


def test_normal_operator():
    assert bench_normal_operator()["synthetic_normal_operator"] == 1.0


def test_multiplication_op():
    assert bench_multiplication_op()["synthetic_multiplication_op"] == 1.0
