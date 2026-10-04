"""Wave-1043 forestry canon tests."""

from __future__ import annotations

from quant_fund.models.dendrology import bench_dendrology
from quant_fund.models.forest_ecology import bench_forest_ecology
from quant_fund.models.forest_economics import bench_forest_economics
from quant_fund.models.silviculture import bench_silviculture
from quant_fund.models.timber_harvesting import bench_timber_harvesting
from quant_fund.models.wildfire_management import bench_wildfire_management


def test_silviculture():
    assert bench_silviculture()["synthetic_silviculture"] == 1.0


def test_forest_ecology():
    assert bench_forest_ecology()["synthetic_forest_ecology"] == 1.0


def test_timber_harvesting():
    assert bench_timber_harvesting()["synthetic_timber_harvesting"] == 1.0


def test_forest_economics():
    assert bench_forest_economics()["synthetic_forest_economics"] == 1.0


def test_dendrology():
    assert bench_dendrology()["synthetic_dendrology"] == 1.0


def test_wildfire_management():
    assert bench_wildfire_management()["synthetic_wildfire_management"] == 1.0
