from quant_fund.models.base_change import bench_base_change
from quant_fund.models.constructible import bench_constructible
from quant_fund.models.perverse_sh import bench_perverse_sh
from quant_fund.models.projection_frm import bench_projection_frm
from quant_fund.models.six_functors import bench_six_functors
from quant_fund.models.verdier_dual import bench_verdier_dual


def test_six_functors():
    assert bench_six_functors()["synthetic_six_functors"] == 1.0


def test_base_change():
    assert bench_base_change()["synthetic_base_change"] == 1.0


def test_projection_frm():
    assert bench_projection_frm()["synthetic_projection_frm"] == 1.0


def test_verdier_dual():
    assert bench_verdier_dual()["synthetic_verdier_dual"] == 1.0


def test_constructible():
    assert bench_constructible()["synthetic_constructible"] == 1.0


def test_perverse_sh():
    assert bench_perverse_sh()["synthetic_perverse_sh"] == 1.0
