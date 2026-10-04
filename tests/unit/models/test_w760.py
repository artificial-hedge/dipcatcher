from quant_fund.models.cameron_martin import bench_cameron_martin
from quant_fund.models.doob_bm import bench_doob_bm
from quant_fund.models.gikhman_skorokhod import (
    bench_gikhman_skorokhod,
)
from quant_fund.models.ito_bm import bench_ito_bm
from quant_fund.models.levy_bm import bench_levy_bm
from quant_fund.models.wiener_bm import bench_wiener_bm


def test_levy_bm():
    assert bench_levy_bm()["synthetic_levy_bm"] == 1.0


def test_wiener_bm():
    assert bench_wiener_bm()["synthetic_wiener_bm"] == 1.0


def test_doob_bm():
    assert bench_doob_bm()["synthetic_doob_bm"] == 1.0


def test_ito_bm():
    assert bench_ito_bm()["synthetic_ito_bm"] == 1.0


def test_cameron_martin():
    assert bench_cameron_martin()["synthetic_cameron_martin"] == 1.0


def test_gikhman_skorokhod():
    assert bench_gikhman_skorokhod()["synthetic_gikhman_skorokhod"] == 1.0
