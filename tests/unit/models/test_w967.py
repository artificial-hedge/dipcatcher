"""Wave-967 C*-dynamics canon tests."""

from __future__ import annotations

from quant_fund.models.crossed_product import bench_crossed_product
from quant_fund.models.cstar_dynamics import bench_cstar_dynamics
from quant_fund.models.kirchberg_absorb import bench_kirchberg_absorb
from quant_fund.models.rokhlin_action import bench_rokhlin_action
from quant_fund.models.taf_dim import bench_taf_dim
from quant_fund.models.z_stability import bench_z_stability


def test_cstar_dynamics():
    assert bench_cstar_dynamics()["synthetic_cstar_dynamics"] == 1.0


def test_crossed_product():
    assert bench_crossed_product()["synthetic_crossed_product"] == 1.0


def test_rokhlin_action():
    assert bench_rokhlin_action()["synthetic_rokhlin_action"] == 1.0


def test_kirchberg_absorb():
    assert bench_kirchberg_absorb()["synthetic_kirchberg_absorb"] == 1.0


def test_taf_dim():
    assert bench_taf_dim()["synthetic_taf_dim"] == 1.0


def test_z_stability():
    assert bench_z_stability()["synthetic_z_stability"] == 1.0
