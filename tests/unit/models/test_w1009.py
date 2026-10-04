"""Wave-1009 electrodynamics/optics canon tests."""

from __future__ import annotations

from quant_fund.models.dipole_radiation import bench_dipole_radiation
from quant_fund.models.fresnel_eq import bench_fresnel_eq
from quant_fund.models.lorentz_lorenz import bench_lorentz_lorenz
from quant_fund.models.maxwell_equations import bench_maxwell_equations
from quant_fund.models.poynting_vector import bench_poynting_vector
from quant_fund.models.wave_guides import bench_wave_guides


def test_maxwell_equations():
    assert bench_maxwell_equations()["synthetic_maxwell_equations"] == 1.0


def test_poynting_vector():
    assert bench_poynting_vector()["synthetic_poynting_vector"] == 1.0


def test_fresnel_eq():
    assert bench_fresnel_eq()["synthetic_fresnel_eq"] == 1.0


def test_wave_guides():
    assert bench_wave_guides()["synthetic_wave_guides"] == 1.0


def test_dipole_radiation():
    assert bench_dipole_radiation()["synthetic_dipole_radiation"] == 1.0


def test_lorentz_lorenz():
    assert bench_lorentz_lorenz()["synthetic_lorentz_lorenz"] == 1.0
