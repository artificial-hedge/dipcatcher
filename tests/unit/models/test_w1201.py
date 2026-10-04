"""Wave-1201 health-informatics canon tests."""

from __future__ import annotations

from quant_fund.models.biomedical_informatics import bench_biomedical_informatics
from quant_fund.models.clinical_informatics import bench_clinical_informatics
from quant_fund.models.health_data_science import bench_health_data_science
from quant_fund.models.health_informatics import bench_health_informatics
from quant_fund.models.health_information import bench_health_information
from quant_fund.models.medical_records import bench_medical_records


def test_health_informatics():
    assert bench_health_informatics()["synthetic_health_informatics"] == 1.0


def test_medical_records():
    assert bench_medical_records()["synthetic_medical_records"] == 1.0


def test_health_information():
    assert bench_health_information()["synthetic_health_information"] == 1.0


def test_biomedical_informatics():
    assert bench_biomedical_informatics()["synthetic_biomedical_informatics"] == 1.0


def test_clinical_informatics():
    assert bench_clinical_informatics()["synthetic_clinical_informatics"] == 1.0


def test_health_data_science():
    assert bench_health_data_science()["synthetic_health_data_science"] == 1.0
