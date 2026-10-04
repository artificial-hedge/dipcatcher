"""Wave-1037 agriculture canon tests."""

from __future__ import annotations

from quant_fund.models.agronomy import bench_agronomy
from quant_fund.models.animal_science import bench_animal_science
from quant_fund.models.crop_science import bench_crop_science
from quant_fund.models.horticulture import bench_horticulture
from quant_fund.models.pest_management import bench_pest_management
from quant_fund.models.soil_science import bench_soil_science


def test_crop_science():
    assert bench_crop_science()["synthetic_crop_science"] == 1.0


def test_soil_science():
    assert bench_soil_science()["synthetic_soil_science"] == 1.0


def test_agronomy():
    assert bench_agronomy()["synthetic_agronomy"] == 1.0


def test_animal_science():
    assert bench_animal_science()["synthetic_animal_science"] == 1.0


def test_horticulture():
    assert bench_horticulture()["synthetic_horticulture"] == 1.0


def test_pest_management():
    assert bench_pest_management()["synthetic_pest_management"] == 1.0
