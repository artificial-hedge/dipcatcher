"""Wave-1114 physics-3 canon tests."""

from __future__ import annotations

from quant_fund.models.classical_mechanics import bench_classical_mechanics
from quant_fund.models.condensed_matter_2 import bench_condensed_matter_2
from quant_fund.models.nuclear_physics import bench_nuclear_physics
from quant_fund.models.plasma_physics import bench_plasma_physics
from quant_fund.models.quantum_mechanics_2 import bench_quantum_mechanics_2
from quant_fund.models.statistical_mechanics_2 import bench_statistical_mechanics_2


def test_classical_mechanics():
    assert bench_classical_mechanics()["synthetic_classical_mechanics"] == 1.0


def test_quantum_mechanics_2():
    assert bench_quantum_mechanics_2()["synthetic_quantum_mechanics_2"] == 1.0


def test_statistical_mechanics_2():
    assert bench_statistical_mechanics_2()["synthetic_statistical_mechanics_2"] == 1.0


def test_nuclear_physics():
    assert bench_nuclear_physics()["synthetic_nuclear_physics"] == 1.0


def test_plasma_physics():
    assert bench_plasma_physics()["synthetic_plasma_physics"] == 1.0


def test_condensed_matter_2():
    assert bench_condensed_matter_2()["synthetic_condensed_matter_2"] == 1.0
