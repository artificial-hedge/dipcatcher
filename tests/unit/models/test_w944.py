"""Wave-944 matrix-analysis-2 canon tests."""

from __future__ import annotations

from quant_fund.models.cauchy_binet import bench_cauchy_binet
from quant_fund.models.fan_inequality import bench_fan_inequality
from quant_fund.models.horn_inequality import bench_horn_inequality
from quant_fund.models.majorization_vec import bench_majorization_vec
from quant_fund.models.schur_complement import bench_schur_complement
from quant_fund.models.weyl_ineq import bench_weyl_ineq


def test_fan_inequality():
    assert bench_fan_inequality()["synthetic_fan_inequality"] == 1.0


def test_horn_inequality():
    assert bench_horn_inequality()["synthetic_horn_inequality"] == 1.0


def test_weyl_ineq():
    assert bench_weyl_ineq()["synthetic_weyl_ineq"] == 1.0


def test_cauchy_binet():
    assert bench_cauchy_binet()["synthetic_cauchy_binet"] == 1.0


def test_schur_complement():
    assert bench_schur_complement()["synthetic_schur_complement"] == 1.0


def test_majorization_vec():
    assert bench_majorization_vec()["synthetic_majorization_vec"] == 1.0
