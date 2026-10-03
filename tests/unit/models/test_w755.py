from quant_fund.models.barlow_ust import bench_barlow_ust
from quant_fund.models.kassel_wu import bench_kassel_wu
from quant_fund.models.kenyon_wilson import bench_kenyon_wilson
from quant_fund.models.lejan_loop import bench_lejan_loop
from quant_fund.models.lupu_loop import bench_lupu_loop
from quant_fund.models.lyons_peres import bench_lyons_peres


def test_lupu_loop():
    assert bench_lupu_loop()["synthetic_lupu_loop"] == 1.0


def test_lejan_loop():
    assert bench_lejan_loop()["synthetic_lejan_loop"] == 1.0


def test_kassel_wu():
    assert bench_kassel_wu()["synthetic_kassel_wu"] == 1.0


def test_kenyon_wilson():
    assert bench_kenyon_wilson()["synthetic_kenyon_wilson"] == 1.0


def test_barlow_ust():
    assert bench_barlow_ust()["synthetic_barlow_ust"] == 1.0


def test_lyons_peres():
    assert bench_lyons_peres()["synthetic_lyons_peres"] == 1.0
