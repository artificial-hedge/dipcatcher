"""Wave-1154 fundamental-physics canon tests."""

from __future__ import annotations

from quant_fund.models.electromagnetism import bench_electromagnetism
from quant_fund.models.nuclear_physics_2 import bench_nuclear_physics_2
from quant_fund.models.optics_4 import bench_optics_4
from quant_fund.models.particle_physics import bench_particle_physics
from quant_fund.models.quantum_physics import bench_quantum_physics
from quant_fund.models.relativity_3 import bench_relativity_3


def test_electromagnetism():
    assert bench_electromagnetism()["synthetic_electromagnetism"] == 1.0


def test_optics_4():
    assert bench_optics_4()["synthetic_optics_4"] == 1.0


def test_nuclear_physics_2():
    assert bench_nuclear_physics_2()["synthetic_nuclear_physics_2"] == 1.0


def test_particle_physics():
    assert bench_particle_physics()["synthetic_particle_physics"] == 1.0


def test_quantum_physics():
    assert bench_quantum_physics()["synthetic_quantum_physics"] == 1.0


def test_relativity_3():
    assert bench_relativity_3()["synthetic_relativity_3"] == 1.0
