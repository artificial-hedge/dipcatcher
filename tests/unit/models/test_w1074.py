"""Wave-1074 culinary arts canon tests."""

from __future__ import annotations

from quant_fund.models.baking_science import bench_baking_science
from quant_fund.models.culinary_arts import bench_culinary_arts
from quant_fund.models.fermentation_science import bench_fermentation_science
from quant_fund.models.flavor_science import bench_flavor_science
from quant_fund.models.food_studies import bench_food_studies
from quant_fund.models.gastronomy import bench_gastronomy


def test_culinary_arts():
    assert bench_culinary_arts()["synthetic_culinary_arts"] == 1.0


def test_gastronomy():
    assert bench_gastronomy()["synthetic_gastronomy"] == 1.0


def test_food_studies():
    assert bench_food_studies()["synthetic_food_studies"] == 1.0


def test_baking_science():
    assert bench_baking_science()["synthetic_baking_science"] == 1.0


def test_flavor_science():
    assert bench_flavor_science()["synthetic_flavor_science"] == 1.0


def test_fermentation_science():
    assert bench_fermentation_science()["synthetic_fermentation_science"] == 1.0
