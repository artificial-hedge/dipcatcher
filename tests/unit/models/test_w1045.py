"""Wave-1045 geology canon tests."""

from __future__ import annotations

from quant_fund.models.geochemistry import bench_geochemistry
from quant_fund.models.geochronology import bench_geochronology
from quant_fund.models.paleontology import bench_paleontology
from quant_fund.models.petrology import bench_petrology
from quant_fund.models.stratigraphy import bench_stratigraphy
from quant_fund.models.structural_geology import bench_structural_geology


def test_stratigraphy():
    assert bench_stratigraphy()["synthetic_stratigraphy"] == 1.0


def test_structural_geology():
    assert bench_structural_geology()["synthetic_structural_geology"] == 1.0


def test_petrology():
    assert bench_petrology()["synthetic_petrology"] == 1.0


def test_geochemistry():
    assert bench_geochemistry()["synthetic_geochemistry"] == 1.0


def test_geochronology():
    assert bench_geochronology()["synthetic_geochronology"] == 1.0


def test_paleontology():
    assert bench_paleontology()["synthetic_paleontology"] == 1.0
