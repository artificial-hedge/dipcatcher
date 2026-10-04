"""Wave-1125 economics-4 canon tests."""

from __future__ import annotations

from quant_fund.models.agricultural_economics import bench_agricultural_economics
from quant_fund.models.development_economics import bench_development_economics
from quant_fund.models.energy_economics import bench_energy_economics
from quant_fund.models.environmental_economics import bench_environmental_economics
from quant_fund.models.health_economics import bench_health_economics
from quant_fund.models.urban_economics import bench_urban_economics


def test_development_economics():
    assert bench_development_economics()["synthetic_development_economics"] == 1.0


def test_environmental_economics():
    assert bench_environmental_economics()["synthetic_environmental_economics"] == 1.0


def test_health_economics():
    assert bench_health_economics()["synthetic_health_economics"] == 1.0


def test_urban_economics():
    assert bench_urban_economics()["synthetic_urban_economics"] == 1.0


def test_agricultural_economics():
    assert bench_agricultural_economics()["synthetic_agricultural_economics"] == 1.0


def test_energy_economics():
    assert bench_energy_economics()["synthetic_energy_economics"] == 1.0
