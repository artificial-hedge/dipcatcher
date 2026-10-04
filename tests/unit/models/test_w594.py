from quant_fund.models.artin_neighborhood import (
    bench_artin_neighborhood,
)
from quant_fund.models.etale_fund import bench_etale_fund
from quant_fund.models.etale_homotopy import (
    bench_etale_homotopy,
)
from quant_fund.models.galois_cat import bench_galois_cat
from quant_fund.models.pro_etale import bench_pro_etale
from quant_fund.models.shapiro_lemma import (
    bench_shapiro_lemma,
)


def test_etale_homotopy():
    assert bench_etale_homotopy()["synthetic_etale_homotopy"] == 1.0


def test_pro_etale():
    assert bench_pro_etale()["synthetic_pro_etale"] == 1.0


def test_etale_fund():
    assert bench_etale_fund()["synthetic_etale_fund"] == 1.0


def test_galois_cat():
    assert bench_galois_cat()["synthetic_galois_cat"] == 1.0


def test_artin_neighborhood():
    assert bench_artin_neighborhood()["synthetic_artin_neighborhood"] == 1.0


def test_shapiro_lemma():
    assert bench_shapiro_lemma()["synthetic_shapiro_lemma"] == 1.0
