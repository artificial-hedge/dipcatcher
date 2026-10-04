from quant_fund.models.bessel3 import bench_bessel3
from quant_fund.models.h_transform import bench_h_transform
from quant_fund.models.occupation_bm import bench_occupation_bm
from quant_fund.models.ost_calcul import bench_ost_calcul
from quant_fund.models.reflect_bm import bench_reflect_bm
from quant_fund.models.tanaka import bench_tanaka


def test_ost_calcul():
    assert bench_ost_calcul()["synthetic_ost_calcul"] == 1.0


def test_tanaka():
    assert bench_tanaka()["synthetic_tanaka"] == 1.0


def test_bessel3():
    assert bench_bessel3()["synthetic_bessel3"] == 1.0


def test_reflect_bm():
    assert bench_reflect_bm()["synthetic_reflect_bm"] == 1.0


def test_occupation_bm():
    assert bench_occupation_bm()["synthetic_occupation_bm"] == 1.0


def test_h_transform():
    assert bench_h_transform()["synthetic_h_transform"] == 1.0
