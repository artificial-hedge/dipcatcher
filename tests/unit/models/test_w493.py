from quant_fund.models.berkovich_an import bench_berkovich_an
from quant_fund.models.mikhalkin import bench_mikhalkin
from quant_fund.models.skeleton_trop import bench_skeleton_trop
from quant_fund.models.tropical_curve import bench_tropical_curve
from quant_fund.models.tropical_cycle import bench_tropical_cycle
from quant_fund.models.tropical_poly import bench_tropical_poly


def test_tropical_poly():
    assert bench_tropical_poly()["synthetic_tropical_poly"] == 1.0


def test_berkovich_an():
    assert bench_berkovich_an()["synthetic_berkovich_an"] == 1.0


def test_skeleton_trop():
    assert bench_skeleton_trop()["synthetic_skeleton_trop"] == 1.0


def test_tropical_curve():
    assert bench_tropical_curve()["synthetic_tropical_curve"] == 1.0


def test_mikhalkin():
    assert bench_mikhalkin()["synthetic_mikhalkin"] == 1.0


def test_tropical_cycle():
    assert bench_tropical_cycle()["synthetic_tropical_cycle"] == 1.0
