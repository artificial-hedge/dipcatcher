"""Wave-1226 pathology canon tests."""

from __future__ import annotations

from quant_fund.models.anatomical_pathology import bench_anatomical_pathology
from quant_fund.models.clinical_pathology import bench_clinical_pathology
from quant_fund.models.cytopathology import bench_cytopathology
from quant_fund.models.histopathology_studies import bench_histopathology_studies
from quant_fund.models.molecular_pathology import bench_molecular_pathology
from quant_fund.models.pathology_studies import bench_pathology_studies


def test_pathology_studies():
    assert bench_pathology_studies()["synthetic_pathology_studies"] == 1.0


def test_anatomical_pathology():
    assert bench_anatomical_pathology()["synthetic_anatomical_pathology"] == 1.0


def test_clinical_pathology():
    assert bench_clinical_pathology()["synthetic_clinical_pathology"] == 1.0


def test_histopathology_studies():
    assert bench_histopathology_studies()["synthetic_histopathology_studies"] == 1.0


def test_cytopathology():
    assert bench_cytopathology()["synthetic_cytopathology"] == 1.0


def test_molecular_pathology():
    assert bench_molecular_pathology()["synthetic_molecular_pathology"] == 1.0
