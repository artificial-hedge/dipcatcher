"""Wave-1027 materials-science canon tests."""

from __future__ import annotations

from quant_fund.models.ceramics import bench_ceramics
from quant_fund.models.crystal_structure import bench_crystal_structure
from quant_fund.models.metallurgy import bench_metallurgy
from quant_fund.models.nanomaterials import bench_nanomaterials
from quant_fund.models.polymer_physics import bench_polymer_physics
from quant_fund.models.superconductivity import bench_superconductivity


def test_crystal_structure():
    assert bench_crystal_structure()["synthetic_crystal_structure"] == 1.0


def test_polymer_physics():
    assert bench_polymer_physics()["synthetic_polymer_physics"] == 1.0


def test_metallurgy():
    assert bench_metallurgy()["synthetic_metallurgy"] == 1.0


def test_ceramics():
    assert bench_ceramics()["synthetic_ceramics"] == 1.0


def test_nanomaterials():
    assert bench_nanomaterials()["synthetic_nanomaterials"] == 1.0


def test_superconductivity():
    assert bench_superconductivity()["synthetic_superconductivity"] == 1.0
