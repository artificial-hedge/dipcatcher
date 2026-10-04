"""Wave-1031 civil-engineering canon tests."""

from __future__ import annotations

from quant_fund.models.construction_mgmt import bench_construction_mgmt
from quant_fund.models.geotechnics import bench_geotechnics
from quant_fund.models.structural_analysis import bench_structural_analysis
from quant_fund.models.surveying import bench_surveying
from quant_fund.models.transportation_eng import bench_transportation_eng
from quant_fund.models.water_resources import bench_water_resources


def test_structural_analysis():
    assert bench_structural_analysis()["synthetic_structural_analysis"] == 1.0


def test_geotechnics():
    assert bench_geotechnics()["synthetic_geotechnics"] == 1.0


def test_transportation_eng():
    assert bench_transportation_eng()["synthetic_transportation_eng"] == 1.0


def test_water_resources():
    assert bench_water_resources()["synthetic_water_resources"] == 1.0


def test_construction_mgmt():
    assert bench_construction_mgmt()["synthetic_construction_mgmt"] == 1.0


def test_surveying():
    assert bench_surveying()["synthetic_surveying"] == 1.0
