from quant_fund.models.cat_lax import bench_cat_lax
from quant_fund.models.cat_pushout import bench_cat_pushout
from quant_fund.models.cat_size import bench_cat_size
from quant_fund.models.cat_span import bench_cat_span
from quant_fund.models.cat_street import bench_cat_street
from quant_fund.models.cat_total import bench_cat_total


def test_cat_pushout():
    assert bench_cat_pushout()["synthetic_cat_pushout"] == 1.0


def test_cat_span():
    assert bench_cat_span()["synthetic_cat_span"] == 1.0


def test_cat_lax():
    assert bench_cat_lax()["synthetic_cat_lax"] == 1.0


def test_cat_street():
    assert bench_cat_street()["synthetic_cat_street"] == 1.0


def test_cat_size():
    assert bench_cat_size()["synthetic_cat_size"] == 1.0


def test_cat_total():
    assert bench_cat_total()["synthetic_cat_total"] == 1.0
