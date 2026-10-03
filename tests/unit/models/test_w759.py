from quant_fund.models.apu_cle import bench_apu_cle
from quant_fund.models.gwynne_cle import bench_gwynne_cle
from quant_fund.models.hospitsky_cle import bench_hospitsky_cle
from quant_fund.models.nolin_cle import bench_nolin_cle
from quant_fund.models.sun_cle import bench_sun_cle
from quant_fund.models.zhan_cle import bench_zhan_cle


def test_gwynne_cle():
    assert bench_gwynne_cle()["synthetic_gwynne_cle"] == 1.0


def test_hospitsky_cle():
    assert bench_hospitsky_cle()["synthetic_hospitsky_cle"] == 1.0


def test_apu_cle():
    assert bench_apu_cle()["synthetic_apu_cle"] == 1.0


def test_nolin_cle():
    assert bench_nolin_cle()["synthetic_nolin_cle"] == 1.0


def test_sun_cle():
    assert bench_sun_cle()["synthetic_sun_cle"] == 1.0


def test_zhan_cle():
    assert bench_zhan_cle()["synthetic_zhan_cle"] == 1.0
