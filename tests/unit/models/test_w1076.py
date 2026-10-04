"""Wave-1076 archaeology canon tests."""

from __future__ import annotations

from quant_fund.models.archaeometry import bench_archaeometry
from quant_fund.models.bioarchaeology import bench_bioarchaeology
from quant_fund.models.experimental_archaeology import bench_experimental_archaeology
from quant_fund.models.field_archaeology import bench_field_archaeology
from quant_fund.models.landscape_archaeology import bench_landscape_archaeology
from quant_fund.models.underwater_archaeology import bench_underwater_archaeology


def test_field_archaeology():
    assert bench_field_archaeology()["synthetic_field_archaeology"] == 1.0


def test_archaeometry():
    assert bench_archaeometry()["synthetic_archaeometry"] == 1.0


def test_bioarchaeology():
    assert bench_bioarchaeology()["synthetic_bioarchaeology"] == 1.0


def test_underwater_archaeology():
    assert bench_underwater_archaeology()["synthetic_underwater_archaeology"] == 1.0


def test_landscape_archaeology():
    assert bench_landscape_archaeology()["synthetic_landscape_archaeology"] == 1.0


def test_experimental_archaeology():
    assert bench_experimental_archaeology()["synthetic_experimental_archaeology"] == 1.0
