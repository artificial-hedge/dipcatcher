"""Wave-958 matrix-inequalities canon tests."""

from __future__ import annotations

from quant_fund.models.araki_lieb_thirring import bench_araki_lieb_thirring
from quant_fund.models.hadamard_fischer import bench_hadamard_fischer
from quant_fund.models.ky_fan import bench_ky_fan
from quant_fund.models.lidskii_thm import bench_lidskii_thm
from quant_fund.models.pinching_ineq import bench_pinching_ineq
from quant_fund.models.von_neumann_trace import bench_von_neumann_trace


def test_ky_fan():
    assert bench_ky_fan()["synthetic_ky_fan"] == 1.0


def test_lidskii_thm():
    assert bench_lidskii_thm()["synthetic_lidskii_thm"] == 1.0


def test_von_neumann_trace():
    assert bench_von_neumann_trace()["synthetic_von_neumann_trace"] == 1.0


def test_pinching_ineq():
    assert bench_pinching_ineq()["synthetic_pinching_ineq"] == 1.0


def test_araki_lieb_thirring():
    assert bench_araki_lieb_thirring()["synthetic_araki_lieb_thirring"] == 1.0


def test_hadamard_fischer():
    assert bench_hadamard_fischer()["synthetic_hadamard_fischer"] == 1.0
