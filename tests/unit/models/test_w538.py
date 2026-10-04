from quant_fund.models.degiorgi_nash import bench_degiorgi_nash
from quant_fund.models.harnack_thm import bench_harnack_thm
from quant_fund.models.poincare_ineq import bench_poincare_ineq
from quant_fund.models.schauder_est import bench_schauder_est
from quant_fund.models.sobolev_space import bench_sobolev_space
from quant_fund.models.trace_thm import bench_trace_thm


def test_sobolev_space():
    assert bench_sobolev_space()["synthetic_sobolev_space"] == 1.0


def test_poincare_ineq():
    assert bench_poincare_ineq()["synthetic_poincare_ineq"] == 1.0


def test_trace_thm():
    assert bench_trace_thm()["synthetic_trace_thm"] == 1.0


def test_harnack_thm():
    assert bench_harnack_thm()["synthetic_harnack_thm"] == 1.0


def test_schauder_est():
    assert bench_schauder_est()["synthetic_schauder_est"] == 1.0


def test_degiorgi_nash():
    assert bench_degiorgi_nash()["synthetic_degiorgi_nash"] == 1.0
