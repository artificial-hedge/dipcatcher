"""Wave-1219 hem-onc canon tests."""

from __future__ import annotations

from quant_fund.models.hematologic_malignancies import bench_hematologic_malignancies
from quant_fund.models.hematology_studies import bench_hematology_studies
from quant_fund.models.oncology_studies import bench_oncology_studies
from quant_fund.models.radiation_oncology import bench_radiation_oncology
from quant_fund.models.solid_tumor_oncology import bench_solid_tumor_oncology
from quant_fund.models.transfusion_medicine import bench_transfusion_medicine


def test_hematology_studies():
    assert bench_hematology_studies()["synthetic_hematology_studies"] == 1.0


def test_oncology_studies():
    assert bench_oncology_studies()["synthetic_oncology_studies"] == 1.0


def test_hematologic_malignancies():
    assert bench_hematologic_malignancies()["synthetic_hematologic_malignancies"] == 1.0


def test_solid_tumor_oncology():
    assert bench_solid_tumor_oncology()["synthetic_solid_tumor_oncology"] == 1.0


def test_transfusion_medicine():
    assert bench_transfusion_medicine()["synthetic_transfusion_medicine"] == 1.0


def test_radiation_oncology():
    assert bench_radiation_oncology()["synthetic_radiation_oncology"] == 1.0
