"""Wave-1029 mechanical-engineering canon tests."""

from __future__ import annotations

from quant_fund.models.fatigue_life import bench_fatigue_life
from quant_fund.models.kinematics import bench_kinematics
from quant_fund.models.machine_design import bench_machine_design
from quant_fund.models.solid_mechanics import bench_solid_mechanics
from quant_fund.models.tribology import bench_tribology
from quant_fund.models.vibration_analysis import bench_vibration_analysis


def test_solid_mechanics():
    assert bench_solid_mechanics()["synthetic_solid_mechanics"] == 1.0


def test_vibration_analysis():
    assert bench_vibration_analysis()["synthetic_vibration_analysis"] == 1.0


def test_fatigue_life():
    assert bench_fatigue_life()["synthetic_fatigue_life"] == 1.0


def test_tribology():
    assert bench_tribology()["synthetic_tribology"] == 1.0


def test_machine_design():
    assert bench_machine_design()["synthetic_machine_design"] == 1.0


def test_kinematics():
    assert bench_kinematics()["synthetic_kinematics"] == 1.0
