"""Wave-953 operator-theory canon tests."""

from __future__ import annotations

from quant_fund.models.bounded_operator import bench_bounded_operator
from quant_fund.models.isometry_operator import bench_isometry_operator
from quant_fund.models.operator_adjoint import bench_operator_adjoint
from quant_fund.models.operator_norm import bench_operator_norm
from quant_fund.models.positive_operator import bench_positive_operator
from quant_fund.models.projection_operator import bench_projection_operator


def test_bounded_operator():
    assert bench_bounded_operator()["synthetic_bounded_operator"] == 1.0


def test_operator_norm():
    assert bench_operator_norm()["synthetic_operator_norm"] == 1.0


def test_operator_adjoint():
    assert bench_operator_adjoint()["synthetic_operator_adjoint"] == 1.0


def test_projection_operator():
    assert bench_projection_operator()["synthetic_projection_operator"] == 1.0


def test_positive_operator():
    assert bench_positive_operator()["synthetic_positive_operator"] == 1.0


def test_isometry_operator():
    assert bench_isometry_operator()["synthetic_isometry_operator"] == 1.0
