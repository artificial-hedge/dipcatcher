"""Wave-1171 geography canon tests."""

from __future__ import annotations

from quant_fund.models.demography_2 import bench_demography_2
from quant_fund.models.geography_2 import bench_geography_2
from quant_fund.models.gis_science_2 import bench_gis_science_2
from quant_fund.models.land_use import bench_land_use
from quant_fund.models.regional_science import bench_regional_science
from quant_fund.models.urbanization import bench_urbanization


def test_geography_2():
    assert bench_geography_2()["synthetic_geography_2"] == 1.0


def test_regional_science():
    assert bench_regional_science()["synthetic_regional_science"] == 1.0


def test_demography_2():
    assert bench_demography_2()["synthetic_demography_2"] == 1.0


def test_urbanization():
    assert bench_urbanization()["synthetic_urbanization"] == 1.0


def test_land_use():
    assert bench_land_use()["synthetic_land_use"] == 1.0


def test_gis_science_2():
    assert bench_gis_science_2()["synthetic_gis_science_2"] == 1.0
