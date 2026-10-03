from quant_fund.models.e_infty2 import bench_e_infty2
from quant_fund.models.h_space import bench_h_space
from quant_fund.models.james_constr import bench_james_constr
from quant_fund.models.obstruction_th import bench_obstruction_th
from quant_fund.models.power_op import bench_power_op
from quant_fund.models.rational_htpy import bench_rational_htpy


def test_e_infty2():
    assert bench_e_infty2()["synthetic_e_infty2"] == 1.0


def test_power_op():
    assert bench_power_op()["synthetic_power_op"] == 1.0


def test_obstruction_th():
    assert bench_obstruction_th()["synthetic_obstruction_th"] == 1.0


def test_rational_htpy():
    assert bench_rational_htpy()["synthetic_rational_htpy"] == 1.0


def test_h_space():
    assert bench_h_space()["synthetic_h_space"] == 1.0


def test_james_constr():
    assert bench_james_constr()["synthetic_james_constr"] == 1.0
