"""Wave-1165 agriculture canon tests."""

from __future__ import annotations

from quant_fund.models.agriculture_2 import bench_agriculture_2
from quant_fund.models.fisheries_2 import bench_fisheries_2
from quant_fund.models.food_science_2 import bench_food_science_2
from quant_fund.models.forestry_2 import bench_forestry_2
from quant_fund.models.horticulture_2 import bench_horticulture_2
from quant_fund.models.veterinary_science_2 import bench_veterinary_science_2


def test_agriculture_2():
    assert bench_agriculture_2()["synthetic_agriculture_2"] == 1.0


def test_food_science_2():
    assert bench_food_science_2()["synthetic_food_science_2"] == 1.0


def test_forestry_2():
    assert bench_forestry_2()["synthetic_forestry_2"] == 1.0


def test_fisheries_2():
    assert bench_fisheries_2()["synthetic_fisheries_2"] == 1.0


def test_horticulture_2():
    assert bench_horticulture_2()["synthetic_horticulture_2"] == 1.0


def test_veterinary_science_2():
    assert bench_veterinary_science_2()["synthetic_veterinary_science_2"] == 1.0
