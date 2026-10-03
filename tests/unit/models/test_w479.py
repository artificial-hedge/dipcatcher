from quant_fund.models.ahb_ring import bench_ahb_ring
from quant_fund.models.drinfeld_sym import bench_drinfeld_sym
from quant_fund.models.fargues_diam import bench_fargues_diam
from quant_fund.models.prism_2 import bench_prism_2
from quant_fund.models.scholze_diamond import bench_scholze_diamond
from quant_fund.models.tilting_equiv import bench_tilting_equiv


def test_fargues_diam():
    assert bench_fargues_diam()["synthetic_fargues_diam"] == 1.0


def test_tilting_equiv():
    assert bench_tilting_equiv()["synthetic_tilting_equiv"] == 1.0


def test_scholze_diamond():
    assert bench_scholze_diamond()["synthetic_scholze_diamond"] == 1.0


def test_ahb_ring():
    assert bench_ahb_ring()["synthetic_ahb_ring"] == 1.0


def test_prism_2():
    assert bench_prism_2()["synthetic_prism_2"] == 1.0


def test_drinfeld_sym():
    assert bench_drinfeld_sym()["synthetic_drinfeld_sym"] == 1.0
