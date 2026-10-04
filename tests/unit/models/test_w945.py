"""Wave-945 matrix-norm canon tests."""

from __future__ import annotations

from quant_fund.models.hankel_op import bench_hankel_op
from quant_fund.models.kyfan_norm import bench_kyfan_norm
from quant_fund.models.matrix_det import bench_matrix_det
from quant_fund.models.numerical_radius import bench_numerical_radius
from quant_fund.models.pfaffian_poly import bench_pfaffian_poly
from quant_fund.models.schatten_norm import bench_schatten_norm


def test_kyfan_norm():
    assert bench_kyfan_norm()["synthetic_kyfan_norm"] == 1.0


def test_schatten_norm():
    assert bench_schatten_norm()["synthetic_schatten_norm"] == 1.0


def test_numerical_radius():
    assert bench_numerical_radius()["synthetic_numerical_radius"] == 1.0


def test_matrix_det():
    assert bench_matrix_det()["synthetic_matrix_det"] == 1.0


def test_pfaffian_poly():
    assert bench_pfaffian_poly()["synthetic_pfaffian_poly"] == 1.0


def test_hankel_op():
    assert bench_hankel_op()["synthetic_hankel_op"] == 1.0
