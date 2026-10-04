"""Wave-1188 culinary canon tests."""

from __future__ import annotations

from quant_fund.models.brewing_science import bench_brewing_science
from quant_fund.models.culinary_science import bench_culinary_science
from quant_fund.models.enology import bench_enology
from quant_fund.models.fermentation_studies import bench_fermentation_studies
from quant_fund.models.gastronomy_2 import bench_gastronomy_2
from quant_fund.models.pastry_arts import bench_pastry_arts


def test_culinary_science():
    assert bench_culinary_science()["synthetic_culinary_science"] == 1.0


def test_pastry_arts():
    assert bench_pastry_arts()["synthetic_pastry_arts"] == 1.0


def test_brewing_science():
    assert bench_brewing_science()["synthetic_brewing_science"] == 1.0


def test_enology():
    assert bench_enology()["synthetic_enology"] == 1.0


def test_fermentation_studies():
    assert bench_fermentation_studies()["synthetic_fermentation_studies"] == 1.0


def test_gastronomy_2():
    assert bench_gastronomy_2()["synthetic_gastronomy_2"] == 1.0
