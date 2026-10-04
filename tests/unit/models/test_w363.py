from quant_fund.models.baire_category import bench_baire_category
from quant_fund.models.cantor_set import bench_cantor_set
from quant_fund.models.egorov_thm import bench_egorov_thm
from quant_fund.models.fatou_lemma import bench_fatou_lemma
from quant_fund.models.monotone_conv import bench_monotone_conv
from quant_fund.models.vitali_set import bench_vitali_set


def test_cantor_set():
    assert bench_cantor_set()["synthetic_cantor_set"] == 1.0


def test_baire_category():
    assert bench_baire_category()["synthetic_baire_category"] == 1.0


def test_vitali_set():
    assert bench_vitali_set()["synthetic_vitali_set"] == 1.0


def test_egorov_thm():
    assert bench_egorov_thm()["synthetic_egorov_thm"] == 1.0


def test_fatou_lemma():
    assert bench_fatou_lemma()["synthetic_fatou_lemma"] == 1.0


def test_monotone_conv():
    assert bench_monotone_conv()["synthetic_monotone_conv"] == 1.0
