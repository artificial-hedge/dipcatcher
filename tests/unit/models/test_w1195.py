"""Wave-1195 clinical-support canon tests."""

from __future__ import annotations

from quant_fund.models.clinical_laboratory import bench_clinical_laboratory
from quant_fund.models.medical_imaging_studies import bench_medical_imaging_studies
from quant_fund.models.mortuary_science import bench_mortuary_science
from quant_fund.models.phlebotomy_studies import bench_phlebotomy_studies
from quant_fund.models.sterile_processing import bench_sterile_processing
from quant_fund.models.surgical_technology import bench_surgical_technology


def test_medical_imaging_studies():
    assert bench_medical_imaging_studies()["synthetic_medical_imaging_studies"] == 1.0


def test_clinical_laboratory():
    assert bench_clinical_laboratory()["synthetic_clinical_laboratory"] == 1.0


def test_mortuary_science():
    assert bench_mortuary_science()["synthetic_mortuary_science"] == 1.0


def test_phlebotomy_studies():
    assert bench_phlebotomy_studies()["synthetic_phlebotomy_studies"] == 1.0


def test_surgical_technology():
    assert bench_surgical_technology()["synthetic_surgical_technology"] == 1.0


def test_sterile_processing():
    assert bench_sterile_processing()["synthetic_sterile_processing"] == 1.0
