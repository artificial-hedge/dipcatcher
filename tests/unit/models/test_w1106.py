"""Wave-1106 materials-2 canon tests."""

from __future__ import annotations

from quant_fund.models.biomaterials import bench_biomaterials
from quant_fund.models.characterization_methods import bench_characterization_methods
from quant_fund.models.composite_materials import bench_composite_materials
from quant_fund.models.phase_diagrams import bench_phase_diagrams
from quant_fund.models.semiconductors_materials import bench_semiconductors_materials
from quant_fund.models.thin_films import bench_thin_films


def test_semiconductors_materials():
    assert bench_semiconductors_materials()["synthetic_semiconductors_materials"] == 1.0


def test_composite_materials():
    assert bench_composite_materials()["synthetic_composite_materials"] == 1.0


def test_thin_films():
    assert bench_thin_films()["synthetic_thin_films"] == 1.0


def test_biomaterials():
    assert bench_biomaterials()["synthetic_biomaterials"] == 1.0


def test_phase_diagrams():
    assert bench_phase_diagrams()["synthetic_phase_diagrams"] == 1.0


def test_characterization_methods():
    assert bench_characterization_methods()["synthetic_characterization_methods"] == 1.0
