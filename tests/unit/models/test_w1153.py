"""Wave-1153 physical-sciences canon tests."""

from __future__ import annotations

from quant_fund.models.astrophysics_3 import bench_astrophysics_3
from quant_fund.models.cosmology_3 import bench_cosmology_3
from quant_fund.models.geophysics_3 import bench_geophysics_3
from quant_fund.models.mechanics import bench_mechanics
from quant_fund.models.physics_6 import bench_physics_6
from quant_fund.models.thermodynamics_3 import bench_thermodynamics_3


def test_physics_6():
    assert bench_physics_6()["synthetic_physics_6"] == 1.0


def test_astrophysics_3():
    assert bench_astrophysics_3()["synthetic_astrophysics_3"] == 1.0


def test_cosmology_3():
    assert bench_cosmology_3()["synthetic_cosmology_3"] == 1.0


def test_geophysics_3():
    assert bench_geophysics_3()["synthetic_geophysics_3"] == 1.0


def test_mechanics():
    assert bench_mechanics()["synthetic_mechanics"] == 1.0


def test_thermodynamics_3():
    assert bench_thermodynamics_3()["synthetic_thermodynamics_3"] == 1.0
