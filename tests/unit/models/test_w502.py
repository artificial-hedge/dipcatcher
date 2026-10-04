from quant_fund.models.dg_cat2 import bench_dg_cat2
from quant_fund.models.dg_morita import bench_dg_morita
from quant_fund.models.dg_nerve import bench_dg_nerve
from quant_fund.models.dg_quotient import bench_dg_quotient
from quant_fund.models.drinfeld_quotient import bench_drinfeld_quotient
from quant_fund.models.keller_dg import bench_keller_dg


def test_dg_cat2():
    assert bench_dg_cat2()["synthetic_dg_cat2"] == 1.0


def test_dg_morita():
    assert bench_dg_morita()["synthetic_dg_morita"] == 1.0


def test_dg_quotient():
    assert bench_dg_quotient()["synthetic_dg_quotient"] == 1.0


def test_drinfeld_quotient():
    assert bench_drinfeld_quotient()["synthetic_drinfeld_quotient"] == 1.0


def test_dg_nerve():
    assert bench_dg_nerve()["synthetic_dg_nerve"] == 1.0


def test_keller_dg():
    assert bench_keller_dg()["synthetic_keller_dg"] == 1.0
