"""Wave-1121 geography-3 canon tests."""

from __future__ import annotations

from quant_fund.models.economic_geography import bench_economic_geography
from quant_fund.models.gis_science import bench_gis_science
from quant_fund.models.health_geography import bench_health_geography
from quant_fund.models.political_geography import bench_political_geography
from quant_fund.models.population_geography import bench_population_geography
from quant_fund.models.regional_geography import bench_regional_geography


def test_regional_geography():
    assert bench_regional_geography()["synthetic_regional_geography"] == 1.0


def test_health_geography():
    assert bench_health_geography()["synthetic_health_geography"] == 1.0


def test_population_geography():
    assert bench_population_geography()["synthetic_population_geography"] == 1.0


def test_economic_geography():
    assert bench_economic_geography()["synthetic_economic_geography"] == 1.0


def test_political_geography():
    assert bench_political_geography()["synthetic_political_geography"] == 1.0


def test_gis_science():
    assert bench_gis_science()["synthetic_gis_science"] == 1.0
