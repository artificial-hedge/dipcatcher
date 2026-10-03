"""Wave-1001 fluid-dynamics canon tests."""

from __future__ import annotations

from quant_fund.models.beale_kato_majda import bench_beale_kato_majda
from quant_fund.models.euler_equations import bench_euler_equations
from quant_fund.models.ladyzhenskaya_weak import bench_ladyzhenskaya_weak
from quant_fund.models.leray_theory import bench_leray_theory
from quant_fund.models.navier_stokes import bench_navier_stokes
from quant_fund.models.vorticity_form import bench_vorticity_form


def test_euler_equations():
    assert bench_euler_equations()["synthetic_euler_equations"] == 1.0


def test_navier_stokes():
    assert bench_navier_stokes()["synthetic_navier_stokes"] == 1.0


def test_vorticity_form():
    assert bench_vorticity_form()["synthetic_vorticity_form"] == 1.0


def test_beale_kato_majda():
    assert bench_beale_kato_majda()["synthetic_beale_kato_majda"] == 1.0


def test_ladyzhenskaya_weak():
    assert bench_ladyzhenskaya_weak()["synthetic_ladyzhenskaya_weak"] == 1.0


def test_leray_theory():
    assert bench_leray_theory()["synthetic_leray_theory"] == 1.0
