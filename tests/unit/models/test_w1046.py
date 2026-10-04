"""Wave-1046 meteorology canon tests."""

from __future__ import annotations

from quant_fund.models.atmospheric_dynamics import bench_atmospheric_dynamics
from quant_fund.models.climate_dynamics import bench_climate_dynamics
from quant_fund.models.cloud_physics import bench_cloud_physics
from quant_fund.models.mesoscale_meteorology import bench_mesoscale_meteorology
from quant_fund.models.numerical_weather import bench_numerical_weather
from quant_fund.models.synoptic_meteorology import bench_synoptic_meteorology


def test_atmospheric_dynamics():
    assert bench_atmospheric_dynamics()["synthetic_atmospheric_dynamics"] == 1.0


def test_synoptic_meteorology():
    assert bench_synoptic_meteorology()["synthetic_synoptic_meteorology"] == 1.0


def test_cloud_physics():
    assert bench_cloud_physics()["synthetic_cloud_physics"] == 1.0


def test_numerical_weather():
    assert bench_numerical_weather()["synthetic_numerical_weather"] == 1.0


def test_mesoscale_meteorology():
    assert bench_mesoscale_meteorology()["synthetic_mesoscale_meteorology"] == 1.0


def test_climate_dynamics():
    assert bench_climate_dynamics()["synthetic_climate_dynamics"] == 1.0
