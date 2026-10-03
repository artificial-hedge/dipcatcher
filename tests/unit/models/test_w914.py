"""Wave-914 RK/BVP-methods-2 canon tests."""

from __future__ import annotations

from quant_fund.models.green_function_bvp import bench_green_function_bvp
from quant_fund.models.invariant_imbedding import bench_invariant_imbedding
from quant_fund.models.ralston_rk import bench_ralston_rk
from quant_fund.models.ralston_second import bench_ralston_second
from quant_fund.models.runge_kutta4 import bench_runge_kutta4
from quant_fund.models.verner_rk import bench_verner_rk


def test_ralston_rk():
    assert bench_ralston_rk()["synthetic_ralston_rk"] == 1.0


def test_verner_rk():
    assert bench_verner_rk()["synthetic_verner_rk"] == 1.0


def test_ralston_second():
    assert bench_ralston_second()["synthetic_ralston_second"] == 1.0


def test_runge_kutta4():
    assert bench_runge_kutta4()["synthetic_runge_kutta4"] == 1.0


def test_invariant_imbedding():
    assert bench_invariant_imbedding()["synthetic_invariant_imbedding"] == 1.0


def test_green_function_bvp():
    assert bench_green_function_bvp()["synthetic_green_function_bvp"] == 1.0
