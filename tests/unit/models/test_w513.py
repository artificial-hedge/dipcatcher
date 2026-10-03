from quant_fund.models.categorify import bench_categorify
from quant_fund.models.hecke_cat import bench_hecke_cat
from quant_fund.models.khovanov_hom import bench_khovanov_hom
from quant_fund.models.rasmussen_inv import bench_rasmussen_inv
from quant_fund.models.soergel_bim import bench_soergel_bim
from quant_fund.models.uq_sl2 import bench_uq_sl2


def test_categorify():
    assert bench_categorify()["synthetic_categorify"] == 1.0


def test_khovanov_hom():
    assert bench_khovanov_hom()["synthetic_khovanov_hom"] == 1.0


def test_hecke_cat():
    assert bench_hecke_cat()["synthetic_hecke_cat"] == 1.0


def test_soergel_bim():
    assert bench_soergel_bim()["synthetic_soergel_bim"] == 1.0


def test_rasmussen_inv():
    assert bench_rasmussen_inv()["synthetic_rasmussen_inv"] == 1.0


def test_uq_sl2():
    assert bench_uq_sl2()["synthetic_uq_sl2"] == 1.0
