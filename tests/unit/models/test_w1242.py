"""Wave-1242 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.fracture_studies import bench_fracture_studies
from quant_fund.models.osteoporosis_studies import bench_osteoporosis_studies
from quant_fund.models.physiatry_studies import bench_physiatry_studies
from quant_fund.models.physical_therapy_studies import bench_physical_therapy_studies
from quant_fund.models.rehabilitation_studies import bench_rehabilitation_studies
from quant_fund.models.sports_injury_studies import bench_sports_injury_studies


def test_rehabilitation_studies() -> None:
    assert bench_rehabilitation_studies()["synthetic_rehabilitation_studies"] == 1.0


def test_physical_therapy_studies() -> None:
    assert bench_physical_therapy_studies()["synthetic_physical_therapy_studies"] == 1.0


def test_sports_injury_studies() -> None:
    assert bench_sports_injury_studies()["synthetic_sports_injury_studies"] == 1.0


def test_fracture_studies() -> None:
    assert bench_fracture_studies()["synthetic_fracture_studies"] == 1.0


def test_osteoporosis_studies() -> None:
    assert bench_osteoporosis_studies()["synthetic_osteoporosis_studies"] == 1.0


def test_physiatry_studies() -> None:
    assert bench_physiatry_studies()["synthetic_physiatry_studies"] == 1.0
