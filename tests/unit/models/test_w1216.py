"""Wave-1216 internal-medicine canon tests."""

from __future__ import annotations

from quant_fund.models.critical_care_medicine import bench_critical_care_medicine
from quant_fund.models.gastroenterology_studies import bench_gastroenterology_studies
from quant_fund.models.hepatology_studies import bench_hepatology_studies
from quant_fund.models.hospital_medicine import bench_hospital_medicine
from quant_fund.models.internal_medicine import bench_internal_medicine
from quant_fund.models.pulmonary_medicine import bench_pulmonary_medicine


def test_internal_medicine():
    assert bench_internal_medicine()["synthetic_internal_medicine"] == 1.0


def test_hospital_medicine():
    assert bench_hospital_medicine()["synthetic_hospital_medicine"] == 1.0


def test_critical_care_medicine():
    assert bench_critical_care_medicine()["synthetic_critical_care_medicine"] == 1.0


def test_pulmonary_medicine():
    assert bench_pulmonary_medicine()["synthetic_pulmonary_medicine"] == 1.0


def test_gastroenterology_studies():
    assert bench_gastroenterology_studies()["synthetic_gastroenterology_studies"] == 1.0


def test_hepatology_studies():
    assert bench_hepatology_studies()["synthetic_hepatology_studies"] == 1.0
