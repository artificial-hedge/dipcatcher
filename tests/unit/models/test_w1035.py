"""Wave-1035 nuclear-engineering canon tests."""

from __future__ import annotations

from quant_fund.models.isotope_production import bench_isotope_production
from quant_fund.models.nuclear_fuel_cycle import bench_nuclear_fuel_cycle
from quant_fund.models.nuclear_safety import bench_nuclear_safety
from quant_fund.models.radiation_protection import bench_radiation_protection
from quant_fund.models.reactor_physics import bench_reactor_physics
from quant_fund.models.thermal_hydraulics import bench_thermal_hydraulics


def test_reactor_physics():
    assert bench_reactor_physics()["synthetic_reactor_physics"] == 1.0


def test_radiation_protection():
    assert bench_radiation_protection()["synthetic_radiation_protection"] == 1.0


def test_nuclear_fuel_cycle():
    assert bench_nuclear_fuel_cycle()["synthetic_nuclear_fuel_cycle"] == 1.0


def test_thermal_hydraulics():
    assert bench_thermal_hydraulics()["synthetic_thermal_hydraulics"] == 1.0


def test_nuclear_safety():
    assert bench_nuclear_safety()["synthetic_nuclear_safety"] == 1.0


def test_isotope_production():
    assert bench_isotope_production()["synthetic_isotope_production"] == 1.0
