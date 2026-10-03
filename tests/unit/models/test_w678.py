from quant_fund.models.equipment_cat import bench_equipment_cat
from quant_fund.models.fibrant_cat import bench_fibrant_cat
from quant_fund.models.homotopical_cat import (
    bench_homotopical_cat,
)
from quant_fund.models.pointed_cat import bench_pointed_cat
from quant_fund.models.relative_cat import bench_relative_cat
from quant_fund.models.simplicial_cat import bench_simplicial_cat


def test_simplicial_cat():
    assert bench_simplicial_cat()["synthetic_simplicial_cat"] == 1.0


def test_homotopical_cat():
    assert bench_homotopical_cat()["synthetic_homotopical_cat"] == 1.0


def test_relative_cat():
    assert bench_relative_cat()["synthetic_relative_cat"] == 1.0


def test_equipment_cat():
    assert bench_equipment_cat()["synthetic_equipment_cat"] == 1.0


def test_fibrant_cat():
    assert bench_fibrant_cat()["synthetic_fibrant_cat"] == 1.0


def test_pointed_cat():
    assert bench_pointed_cat()["synthetic_pointed_cat"] == 1.0
