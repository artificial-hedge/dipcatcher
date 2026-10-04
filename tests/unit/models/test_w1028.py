"""Wave-1028 chemical-engineering canon tests."""

from __future__ import annotations

from quant_fund.models.fluid_dynamics2 import bench_fluid_dynamics2
from quant_fund.models.heat_exchanger import bench_heat_exchanger
from quant_fund.models.process_control import bench_process_control
from quant_fund.models.reaction_kinetics import bench_reaction_kinetics
from quant_fund.models.separation_proc import bench_separation_proc
from quant_fund.models.thermo_props import bench_thermo_props


def test_reaction_kinetics():
    assert bench_reaction_kinetics()["synthetic_reaction_kinetics"] == 1.0


def test_thermo_props():
    assert bench_thermo_props()["synthetic_thermo_props"] == 1.0


def test_separation_proc():
    assert bench_separation_proc()["synthetic_separation_proc"] == 1.0


def test_heat_exchanger():
    assert bench_heat_exchanger()["synthetic_heat_exchanger"] == 1.0


def test_fluid_dynamics2():
    assert bench_fluid_dynamics2()["synthetic_fluid_dynamics2"] == 1.0


def test_process_control():
    assert bench_process_control()["synthetic_process_control"] == 1.0
