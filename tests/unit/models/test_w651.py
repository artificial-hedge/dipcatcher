from quant_fund.models.abelian_cat import bench_abelian_cat
from quant_fund.models.filtered_cat import bench_filtered_cat
from quant_fund.models.flat_functor import bench_flat_functor
from quant_fund.models.malcev_cat import bench_malcev_cat
from quant_fund.models.regular_cat import bench_regular_cat
from quant_fund.models.sifted_cat2 import bench_sifted_cat2


def test_flat_functor():
    assert bench_flat_functor()["synthetic_flat_functor"] == 1.0


def test_filtered_cat():
    assert bench_filtered_cat()["synthetic_filtered_cat"] == 1.0


def test_sifted_cat2():
    assert bench_sifted_cat2()["synthetic_sifted_cat2"] == 1.0


def test_regular_cat():
    assert bench_regular_cat()["synthetic_regular_cat"] == 1.0


def test_abelian_cat():
    assert bench_abelian_cat()["synthetic_abelian_cat"] == 1.0


def test_malcev_cat():
    assert bench_malcev_cat()["synthetic_malcev_cat"] == 1.0
