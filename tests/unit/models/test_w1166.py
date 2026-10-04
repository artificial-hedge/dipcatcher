"""Wave-1166 design canon tests."""

from __future__ import annotations

from quant_fund.models.architecture_2 import bench_architecture_2
from quant_fund.models.graphic_design_2 import bench_graphic_design_2
from quant_fund.models.industrial_design_2 import bench_industrial_design_2
from quant_fund.models.interior_design_2 import bench_interior_design_2
from quant_fund.models.landscape_architecture_2 import bench_landscape_architecture_2
from quant_fund.models.urban_planning_2 import bench_urban_planning_2


def test_architecture_2():
    assert bench_architecture_2()["synthetic_architecture_2"] == 1.0


def test_urban_planning_2():
    assert bench_urban_planning_2()["synthetic_urban_planning_2"] == 1.0


def test_interior_design_2():
    assert bench_interior_design_2()["synthetic_interior_design_2"] == 1.0


def test_landscape_architecture_2():
    assert bench_landscape_architecture_2()["synthetic_landscape_architecture_2"] == 1.0


def test_industrial_design_2():
    assert bench_industrial_design_2()["synthetic_industrial_design_2"] == 1.0


def test_graphic_design_2():
    assert bench_graphic_design_2()["synthetic_graphic_design_2"] == 1.0
