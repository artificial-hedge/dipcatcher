"""Wave-1000 elasticity canon tests."""

from __future__ import annotations

from quant_fund.models.contact_mechanics import bench_contact_mechanics
from quant_fund.models.fracture_mechanics import bench_fracture_mechanics
from quant_fund.models.homogenized_elasticity import bench_homogenized_elasticity
from quant_fund.models.kirchhoff_plate import bench_kirchhoff_plate
from quant_fund.models.mindlin_reissner import bench_mindlin_reissner
from quant_fund.models.navier_elasticity import bench_navier_elasticity


def test_navier_elasticity():
    assert bench_navier_elasticity()["synthetic_navier_elasticity"] == 1.0


def test_kirchhoff_plate():
    assert bench_kirchhoff_plate()["synthetic_kirchhoff_plate"] == 1.0


def test_mindlin_reissner():
    assert bench_mindlin_reissner()["synthetic_mindlin_reissner"] == 1.0


def test_contact_mechanics():
    assert bench_contact_mechanics()["synthetic_contact_mechanics"] == 1.0


def test_fracture_mechanics():
    assert bench_fracture_mechanics()["synthetic_fracture_mechanics"] == 1.0


def test_homogenized_elasticity():
    assert bench_homogenized_elasticity()["synthetic_homogenized_elasticity"] == 1.0
