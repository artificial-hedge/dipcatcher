from quant_fund.models.cat_bicomplete import bench_cat_bicomplete
from quant_fund.models.cat_cofibrant import bench_cat_cofibrant
from quant_fund.models.cat_descent import bench_cat_descent
from quant_fund.models.cat_fibrant_obj import (
    bench_cat_fibrant_obj,
)
from quant_fund.models.cat_glueable import bench_cat_glueable
from quant_fund.models.cat_univariant import bench_cat_univariant


def test_cat_fibrant_obj():
    assert bench_cat_fibrant_obj()["synthetic_cat_fibrant_obj"] == 1.0


def test_cat_cofibrant():
    assert bench_cat_cofibrant()["synthetic_cat_cofibrant"] == 1.0


def test_cat_bicomplete():
    assert bench_cat_bicomplete()["synthetic_cat_bicomplete"] == 1.0


def test_cat_univariant():
    assert bench_cat_univariant()["synthetic_cat_univariant"] == 1.0


def test_cat_descent():
    assert bench_cat_descent()["synthetic_cat_descent"] == 1.0


def test_cat_glueable():
    assert bench_cat_glueable()["synthetic_cat_glueable"] == 1.0
