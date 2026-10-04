"""Wave-1237 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.clinical_pharmacy_studies import bench_clinical_pharmacy_studies
from quant_fund.models.compounding_pharmacy import bench_compounding_pharmacy
from quant_fund.models.hospital_pharmacy_studies import bench_hospital_pharmacy_studies
from quant_fund.models.medication_therapy_mgmt import bench_medication_therapy_mgmt
from quant_fund.models.pharmacovigilance_studies import bench_pharmacovigilance_studies
from quant_fund.models.pharmacy_practice_studies import bench_pharmacy_practice_studies


def test_clinical_pharmacy_studies() -> None:
    assert bench_clinical_pharmacy_studies()["synthetic_clinical_pharmacy_studies"] == 1.0


def test_pharmacy_practice_studies() -> None:
    assert bench_pharmacy_practice_studies()["synthetic_pharmacy_practice_studies"] == 1.0


def test_medication_therapy_mgmt() -> None:
    assert bench_medication_therapy_mgmt()["synthetic_medication_therapy_mgmt"] == 1.0


def test_compounding_pharmacy() -> None:
    assert bench_compounding_pharmacy()["synthetic_compounding_pharmacy"] == 1.0


def test_pharmacovigilance_studies() -> None:
    assert bench_pharmacovigilance_studies()["synthetic_pharmacovigilance_studies"] == 1.0


def test_hospital_pharmacy_studies() -> None:
    assert bench_hospital_pharmacy_studies()["synthetic_hospital_pharmacy_studies"] == 1.0
