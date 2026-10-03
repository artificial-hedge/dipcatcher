from quant_fund.models.cat_dold_kan import bench_cat_dold_kan
from quant_fund.models.cat_enriched_lim import (
    bench_cat_enriched_lim,
)
from quant_fund.models.cat_hoc import bench_cat_hoc
from quant_fund.models.cat_pseudo_limit import (
    bench_cat_pseudo_limit,
)
from quant_fund.models.cat_reedy_cat import bench_cat_reedy_cat
from quant_fund.models.cat_weak_eq import bench_cat_weak_eq


def test_cat_pseudo_limit():
    assert bench_cat_pseudo_limit()["synthetic_cat_pseudo_limit"] == 1.0


def test_cat_weak_eq():
    assert bench_cat_weak_eq()["synthetic_cat_weak_eq"] == 1.0


def test_cat_reedy_cat():
    assert bench_cat_reedy_cat()["synthetic_cat_reedy_cat"] == 1.0


def test_cat_dold_kan():
    assert bench_cat_dold_kan()["synthetic_cat_dold_kan"] == 1.0


def test_cat_hoc():
    assert bench_cat_hoc()["synthetic_cat_hoc"] == 1.0


def test_cat_enriched_lim():
    assert bench_cat_enriched_lim()["synthetic_cat_enriched_lim"] == 1.0
