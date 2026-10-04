"""Wave-1103 meteorology-2 canon tests."""

from __future__ import annotations

from quant_fund.models.boundary_layer_meteorology import bench_boundary_layer_meteorology
from quant_fund.models.micrometeorology import bench_micrometeorology
from quant_fund.models.polar_meteorology import bench_polar_meteorology
from quant_fund.models.radar_meteorology import bench_radar_meteorology
from quant_fund.models.severe_weather import bench_severe_weather
from quant_fund.models.tropical_meteorology import bench_tropical_meteorology


def test_severe_weather():
    assert bench_severe_weather()["synthetic_severe_weather"] == 1.0


def test_boundary_layer_meteorology():
    assert bench_boundary_layer_meteorology()["synthetic_boundary_layer_meteorology"] == 1.0


def test_radar_meteorology():
    assert bench_radar_meteorology()["synthetic_radar_meteorology"] == 1.0


def test_tropical_meteorology():
    assert bench_tropical_meteorology()["synthetic_tropical_meteorology"] == 1.0


def test_polar_meteorology():
    assert bench_polar_meteorology()["synthetic_polar_meteorology"] == 1.0


def test_micrometeorology():
    assert bench_micrometeorology()["synthetic_micrometeorology"] == 1.0
