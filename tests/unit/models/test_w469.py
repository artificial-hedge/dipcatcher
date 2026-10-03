from quant_fund.models.accessible_cat import bench_accessible_cat
from quant_fund.models.day_conv import bench_day_conv
from quant_fund.models.derivator2 import bench_derivator2
from quant_fund.models.enriched_cat import bench_enriched_cat
from quant_fund.models.fibered_cat import bench_fibered_cat
from quant_fund.models.weight_lim import bench_weight_lim


def test_enriched_cat():
    assert bench_enriched_cat()["synthetic_enriched_cat"] == 1.0


def test_weight_lim():
    assert bench_weight_lim()["synthetic_weight_lim"] == 1.0


def test_fibered_cat():
    assert bench_fibered_cat()["synthetic_fibered_cat"] == 1.0


def test_derivator2():
    assert bench_derivator2()["synthetic_derivator2"] == 1.0


def test_accessible_cat():
    assert bench_accessible_cat()["synthetic_accessible_cat"] == 1.0


def test_day_conv():
    assert bench_day_conv()["synthetic_day_conv"] == 1.0
