"""Wave-1096 environmental-2 canon tests."""

from __future__ import annotations

from quant_fund.models.conservation_biology import bench_conservation_biology
from quant_fund.models.environmental_toxicology import bench_environmental_toxicology
from quant_fund.models.landscape_ecology import bench_landscape_ecology
from quant_fund.models.marine_conservation import bench_marine_conservation
from quant_fund.models.pollution_science import bench_pollution_science
from quant_fund.models.urban_ecology import bench_urban_ecology


def test_pollution_science():
    assert bench_pollution_science()["synthetic_pollution_science"] == 1.0


def test_conservation_biology():
    assert bench_conservation_biology()["synthetic_conservation_biology"] == 1.0


def test_environmental_toxicology():
    assert bench_environmental_toxicology()["synthetic_environmental_toxicology"] == 1.0


def test_urban_ecology():
    assert bench_urban_ecology()["synthetic_urban_ecology"] == 1.0


def test_landscape_ecology():
    assert bench_landscape_ecology()["synthetic_landscape_ecology"] == 1.0


def test_marine_conservation():
    assert bench_marine_conservation()["synthetic_marine_conservation"] == 1.0
