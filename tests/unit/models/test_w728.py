from quant_fund.models.a1_degrees import bench_a1_degrees
from quant_fund.models.emerton_glass import (
    bench_emerton_glass,
)
from quant_fund.models.luan_yao import bench_luan_yao
from quant_fund.models.morel_voev import bench_morel_voev
from quant_fund.models.totaro_cycle import bench_totaro_cycle
from quant_fund.models.voev_homotopy import bench_voev_homotopy


def test_emerton_glass():
    assert bench_emerton_glass()["synthetic_emerton_glass"] == 1.0


def test_luan_yao():
    assert bench_luan_yao()["synthetic_luan_yao"] == 1.0


def test_morel_voev():
    assert bench_morel_voev()["synthetic_morel_voev"] == 1.0


def test_voev_homotopy():
    assert bench_voev_homotopy()["synthetic_voev_homotopy"] == 1.0


def test_totaro_cycle():
    assert bench_totaro_cycle()["synthetic_totaro_cycle"] == 1.0


def test_a1_degrees():
    assert bench_a1_degrees()["synthetic_a1_degrees"] == 1.0
