"""Wave-1041 ocean-engineering canon tests."""

from __future__ import annotations

from quant_fund.models.coastal_engineering import bench_coastal_engineering
from quant_fund.models.marine_propulsion import bench_marine_propulsion
from quant_fund.models.naval_architecture import bench_naval_architecture
from quant_fund.models.ocean_waves import bench_ocean_waves
from quant_fund.models.offshore_engineering import bench_offshore_engineering
from quant_fund.models.submarine_systems import bench_submarine_systems


def test_naval_architecture():
    assert bench_naval_architecture()["synthetic_naval_architecture"] == 1.0


def test_offshore_engineering():
    assert bench_offshore_engineering()["synthetic_offshore_engineering"] == 1.0


def test_marine_propulsion():
    assert bench_marine_propulsion()["synthetic_marine_propulsion"] == 1.0


def test_ocean_waves():
    assert bench_ocean_waves()["synthetic_ocean_waves"] == 1.0


def test_coastal_engineering():
    assert bench_coastal_engineering()["synthetic_coastal_engineering"] == 1.0


def test_submarine_systems():
    assert bench_submarine_systems()["synthetic_submarine_systems"] == 1.0
