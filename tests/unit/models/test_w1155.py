"""Wave-1155 earth-systems canon tests."""

from __future__ import annotations

from quant_fund.models.atmospheric_science import bench_atmospheric_science
from quant_fund.models.earth_system_science import bench_earth_system_science
from quant_fund.models.environmental_science_2 import bench_environmental_science_2
from quant_fund.models.hydrology_3 import bench_hydrology_3
from quant_fund.models.oceanography_2 import bench_oceanography_2
from quant_fund.models.soil_science_2 import bench_soil_science_2


def test_earth_system_science():
    assert bench_earth_system_science()["synthetic_earth_system_science"] == 1.0


def test_oceanography_2():
    assert bench_oceanography_2()["synthetic_oceanography_2"] == 1.0


def test_atmospheric_science():
    assert bench_atmospheric_science()["synthetic_atmospheric_science"] == 1.0


def test_environmental_science_2():
    assert bench_environmental_science_2()["synthetic_environmental_science_2"] == 1.0


def test_soil_science_2():
    assert bench_soil_science_2()["synthetic_soil_science_2"] == 1.0


def test_hydrology_3():
    assert bench_hydrology_3()["synthetic_hydrology_3"] == 1.0
