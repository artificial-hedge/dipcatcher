"""Wave-1032 aerospace-engineering canon tests."""

from __future__ import annotations

from quant_fund.models.aerodynamics import bench_aerodynamics
from quant_fund.models.airfoil_theory import bench_airfoil_theory
from quant_fund.models.flight_dynamics import bench_flight_dynamics
from quant_fund.models.orbital_mechanics2 import bench_orbital_mechanics2
from quant_fund.models.propulsion import bench_propulsion
from quant_fund.models.spacecraft_design import bench_spacecraft_design


def test_aerodynamics():
    assert bench_aerodynamics()["synthetic_aerodynamics"] == 1.0


def test_propulsion():
    assert bench_propulsion()["synthetic_propulsion"] == 1.0


def test_orbital_mechanics2():
    assert bench_orbital_mechanics2()["synthetic_orbital_mechanics2"] == 1.0


def test_flight_dynamics():
    assert bench_flight_dynamics()["synthetic_flight_dynamics"] == 1.0


def test_spacecraft_design():
    assert bench_spacecraft_design()["synthetic_spacecraft_design"] == 1.0


def test_airfoil_theory():
    assert bench_airfoil_theory()["synthetic_airfoil_theory"] == 1.0
