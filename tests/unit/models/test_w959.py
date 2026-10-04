"""Wave-959 operator-theory-3 canon tests."""

from __future__ import annotations

from quant_fund.models.accretive_op import bench_accretive_op
from quant_fund.models.contraction_op import bench_contraction_op
from quant_fund.models.differential_op import bench_differential_op
from quant_fund.models.integral_op import bench_integral_op
from quant_fund.models.sectorial_op import bench_sectorial_op
from quant_fund.models.toeplitz_op import bench_toeplitz_op


def test_toeplitz_op():
    assert bench_toeplitz_op()["synthetic_toeplitz_op"] == 1.0


def test_integral_op():
    assert bench_integral_op()["synthetic_integral_op"] == 1.0


def test_differential_op():
    assert bench_differential_op()["synthetic_differential_op"] == 1.0


def test_contraction_op():
    assert bench_contraction_op()["synthetic_contraction_op"] == 1.0


def test_accretive_op():
    assert bench_accretive_op()["synthetic_accretive_op"] == 1.0


def test_sectorial_op():
    assert bench_sectorial_op()["synthetic_sectorial_op"] == 1.0
