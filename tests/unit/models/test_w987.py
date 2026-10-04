"""Wave-987 parabolic/Li-Yau canon tests."""

from __future__ import annotations

from quant_fund.models.davies_gaffney import bench_davies_gaffney
from quant_fund.models.gaussian_upper import bench_gaussian_upper
from quant_fund.models.grad_est import bench_grad_est
from quant_fund.models.li_yau import bench_li_yau
from quant_fund.models.nash_ineq import bench_nash_ineq
from quant_fund.models.parabolic_harnack import bench_parabolic_harnack


def test_parabolic_harnack():
    assert bench_parabolic_harnack()["synthetic_parabolic_harnack"] == 1.0


def test_gaussian_upper():
    assert bench_gaussian_upper()["synthetic_gaussian_upper"] == 1.0


def test_li_yau():
    assert bench_li_yau()["synthetic_li_yau"] == 1.0


def test_nash_ineq():
    assert bench_nash_ineq()["synthetic_nash_ineq"] == 1.0


def test_davies_gaffney():
    assert bench_davies_gaffney()["synthetic_davies_gaffney"] == 1.0


def test_grad_est():
    assert bench_grad_est()["synthetic_grad_est"] == 1.0
