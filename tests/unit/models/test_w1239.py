"""Wave-1239 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.celiac_studies import bench_celiac_studies
from quant_fund.models.gi_endoscopy_studies import bench_gi_endoscopy_studies
from quant_fund.models.hepatology_medicine import bench_hepatology_medicine
from quant_fund.models.ibd_studies import bench_ibd_studies
from quant_fund.models.motility_studies import bench_motility_studies
from quant_fund.models.pancreatic_medicine import bench_pancreatic_medicine


def test_gi_endoscopy_studies() -> None:
    assert bench_gi_endoscopy_studies()["synthetic_gi_endoscopy_studies"] == 1.0


def test_hepatology_medicine() -> None:
    assert bench_hepatology_medicine()["synthetic_hepatology_medicine"] == 1.0


def test_pancreatic_medicine() -> None:
    assert bench_pancreatic_medicine()["synthetic_pancreatic_medicine"] == 1.0


def test_ibd_studies() -> None:
    assert bench_ibd_studies()["synthetic_ibd_studies"] == 1.0


def test_celiac_studies() -> None:
    assert bench_celiac_studies()["synthetic_celiac_studies"] == 1.0


def test_motility_studies() -> None:
    assert bench_motility_studies()["synthetic_motility_studies"] == 1.0
