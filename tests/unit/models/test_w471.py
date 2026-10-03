from quant_fund.models.banach_colmez import bench_banach_colmez
from quant_fund.models.breuil_kisin import bench_breuil_kisin
from quant_fund.models.diamond_geo import bench_diamond_geo
from quant_fund.models.drinfeld_tower import bench_drinfeld_tower
from quant_fund.models.integral_padic import bench_integral_padic
from quant_fund.models.perfectoid2 import bench_perfectoid2


def test_perfectoid2():
    assert bench_perfectoid2()["synthetic_perfectoid2"] == 1.0


def test_diamond_geo():
    assert bench_diamond_geo()["synthetic_diamond_geo"] == 1.0


def test_integral_padic():
    assert bench_integral_padic()["synthetic_integral_padic"] == 1.0


def test_breuil_kisin():
    assert bench_breuil_kisin()["synthetic_breuil_kisin"] == 1.0


def test_banach_colmez():
    assert bench_banach_colmez()["synthetic_banach_colmez"] == 1.0


def test_drinfeld_tower():
    assert bench_drinfeld_tower()["synthetic_drinfeld_tower"] == 1.0
