"""Wave-1039 environmental-engineering canon tests."""

from __future__ import annotations

from quant_fund.models.air_pollution_control import bench_air_pollution_control
from quant_fund.models.environmental_remediation import bench_environmental_remediation
from quant_fund.models.noise_control import bench_noise_control
from quant_fund.models.waste_management import bench_waste_management
from quant_fund.models.wastewater_engineering import bench_wastewater_engineering
from quant_fund.models.water_treatment import bench_water_treatment


def test_water_treatment():
    assert bench_water_treatment()["synthetic_water_treatment"] == 1.0


def test_air_pollution_control():
    assert bench_air_pollution_control()["synthetic_air_pollution_control"] == 1.0


def test_waste_management():
    assert bench_waste_management()["synthetic_waste_management"] == 1.0


def test_environmental_remediation():
    assert bench_environmental_remediation()["synthetic_environmental_remediation"] == 1.0


def test_wastewater_engineering():
    assert bench_wastewater_engineering()["synthetic_wastewater_engineering"] == 1.0


def test_noise_control():
    assert bench_noise_control()["synthetic_noise_control"] == 1.0
