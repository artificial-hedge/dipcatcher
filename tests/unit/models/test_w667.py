from quant_fund.models.ab_cat import bench_ab_cat
from quant_fund.models.coniveau_fil import bench_coniveau_fil
from quant_fund.models.exact_cat2 import bench_exact_cat2
from quant_fund.models.grothendieck_cat import (
    bench_grothendieck_cat,
)
from quant_fund.models.special_cat import bench_special_cat
from quant_fund.models.stable_cat2 import bench_stable_cat2


def test_stable_cat2():
    assert bench_stable_cat2()["synthetic_stable_cat2"] == 1.0


def test_exact_cat2():
    assert bench_exact_cat2()["synthetic_exact_cat2"] == 1.0


def test_ab_cat():
    assert bench_ab_cat()["synthetic_ab_cat"] == 1.0


def test_grothendieck_cat():
    assert bench_grothendieck_cat()["synthetic_grothendieck_cat"] == 1.0


def test_coniveau_fil():
    assert bench_coniveau_fil()["synthetic_coniveau_fil"] == 1.0


def test_special_cat():
    assert bench_special_cat()["synthetic_special_cat"] == 1.0
