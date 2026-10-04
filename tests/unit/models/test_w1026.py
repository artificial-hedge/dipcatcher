"""Wave-1026 environmental-science canon tests."""

from __future__ import annotations

from quant_fund.models.atmospheric_chem import bench_atmospheric_chem
from quant_fund.models.carbon_cycle import bench_carbon_cycle
from quant_fund.models.climate_model import bench_climate_model
from quant_fund.models.ecosystem_model import bench_ecosystem_model
from quant_fund.models.hydrology import bench_hydrology
from quant_fund.models.ocean_circulation import bench_ocean_circulation


def test_climate_model():
    assert bench_climate_model()["synthetic_climate_model"] == 1.0


def test_ocean_circulation():
    assert bench_ocean_circulation()["synthetic_ocean_circulation"] == 1.0


def test_atmospheric_chem():
    assert bench_atmospheric_chem()["synthetic_atmospheric_chem"] == 1.0


def test_hydrology():
    assert bench_hydrology()["synthetic_hydrology"] == 1.0


def test_carbon_cycle():
    assert bench_carbon_cycle()["synthetic_carbon_cycle"] == 1.0


def test_ecosystem_model():
    assert bench_ecosystem_model()["synthetic_ecosystem_model"] == 1.0
