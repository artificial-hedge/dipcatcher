from quant_fund.models.bicat_comp import bench_bicat_comp
from quant_fund.models.cat_enriched import bench_cat_enriched
from quant_fund.models.double_cat import bench_double_cat
from quant_fund.models.lax_functor import bench_lax_functor
from quant_fund.models.mate_calc import bench_mate_calc
from quant_fund.models.two_cat import bench_two_cat


def test_two_cat():
    assert bench_two_cat()["synthetic_two_cat"] == 1.0


def test_bicat_comp():
    assert bench_bicat_comp()["synthetic_bicat_comp"] == 1.0


def test_mate_calc():
    assert bench_mate_calc()["synthetic_mate_calc"] == 1.0


def test_double_cat():
    assert bench_double_cat()["synthetic_double_cat"] == 1.0


def test_lax_functor():
    assert bench_lax_functor()["synthetic_lax_functor"] == 1.0


def test_cat_enriched():
    assert bench_cat_enriched()["synthetic_cat_enriched"] == 1.0
