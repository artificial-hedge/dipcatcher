from quant_fund.models.diamond_taylor_wiles import (
    bench_diamond_taylor_wiles,
)
from quant_fund.models.jetchev_skinner import (
    bench_jetchev_skinner,
)
from quant_fund.models.kisin_crystalline import (
    bench_kisin_crystalline,
)
from quant_fund.models.mazur_deform import bench_mazur_deform
from quant_fund.models.wan_sss import bench_wan_sss
from quant_fund.models.wiles_taylor import bench_wiles_taylor


def test_jetchev_skinner():
    assert bench_jetchev_skinner()["synthetic_jetchev_skinner"] == 1.0


def test_wan_sss():
    assert bench_wan_sss()["synthetic_wan_sss"] == 1.0


def test_wiles_taylor():
    assert bench_wiles_taylor()["synthetic_wiles_taylor"] == 1.0


def test_diamond_taylor_wiles():
    assert bench_diamond_taylor_wiles()["synthetic_diamond_taylor_wiles"] == 1.0


def test_kisin_crystalline():
    assert bench_kisin_crystalline()["synthetic_kisin_crystalline"] == 1.0


def test_mazur_deform():
    assert bench_mazur_deform()["synthetic_mazur_deform"] == 1.0
