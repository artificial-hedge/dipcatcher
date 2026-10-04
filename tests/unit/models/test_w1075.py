"""Wave-1075 architecture/design canon tests."""

from __future__ import annotations

from quant_fund.models.architecture_theory import bench_architecture_theory
from quant_fund.models.building_science import bench_building_science
from quant_fund.models.industrial_design import bench_industrial_design
from quant_fund.models.interior_design import bench_interior_design
from quant_fund.models.landscape_architecture import bench_landscape_architecture
from quant_fund.models.urban_design import bench_urban_design


def test_architecture_theory():
    assert bench_architecture_theory()["synthetic_architecture_theory"] == 1.0


def test_urban_design():
    assert bench_urban_design()["synthetic_urban_design"] == 1.0


def test_landscape_architecture():
    assert bench_landscape_architecture()["synthetic_landscape_architecture"] == 1.0


def test_interior_design():
    assert bench_interior_design()["synthetic_interior_design"] == 1.0


def test_industrial_design():
    assert bench_industrial_design()["synthetic_industrial_design"] == 1.0


def test_building_science():
    assert bench_building_science()["synthetic_building_science"] == 1.0
