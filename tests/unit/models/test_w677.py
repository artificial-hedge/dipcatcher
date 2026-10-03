from quant_fund.models.cat_dg import bench_cat_dg
from quant_fund.models.cat_structure import bench_cat_structure
from quant_fund.models.combinatorial_mc import (
    bench_combinatorial_mc,
)
from quant_fund.models.derivator_cat import bench_derivator_cat
from quant_fund.models.quillen_cat import bench_quillen_cat
from quant_fund.models.univalent_cat import bench_univalent_cat


def test_derivator_cat():
    assert bench_derivator_cat()["synthetic_derivator_cat"] == 1.0


def test_quillen_cat():
    assert bench_quillen_cat()["synthetic_quillen_cat"] == 1.0


def test_combinatorial_mc():
    assert bench_combinatorial_mc()["synthetic_combinatorial_mc"] == 1.0


def test_cat_dg():
    assert bench_cat_dg()["synthetic_cat_dg"] == 1.0


def test_univalent_cat():
    assert bench_univalent_cat()["synthetic_univalent_cat"] == 1.0


def test_cat_structure():
    assert bench_cat_structure()["synthetic_cat_structure"] == 1.0
