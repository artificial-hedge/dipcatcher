"""Wave-1144 space-science canon tests."""

from __future__ import annotations

from quant_fund.models.asteroid_science import bench_asteroid_science
from quant_fund.models.astrophotonics import bench_astrophotonics
from quant_fund.models.comet_science import bench_comet_science
from quant_fund.models.grav_waves_2 import bench_grav_waves_2
from quant_fund.models.planetology import bench_planetology
from quant_fund.models.space_weather import bench_space_weather


def test_space_weather():
    assert bench_space_weather()["synthetic_space_weather"] == 1.0


def test_planetology():
    assert bench_planetology()["synthetic_planetology"] == 1.0


def test_asteroid_science():
    assert bench_asteroid_science()["synthetic_asteroid_science"] == 1.0


def test_comet_science():
    assert bench_comet_science()["synthetic_comet_science"] == 1.0


def test_astrophotonics():
    assert bench_astrophotonics()["synthetic_astrophotonics"] == 1.0


def test_grav_waves_2():
    assert bench_grav_waves_2()["synthetic_grav_waves_2"] == 1.0
