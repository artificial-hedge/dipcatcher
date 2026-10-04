"""Wave-1233 clinical-genetics canon tests."""

from __future__ import annotations

from quant_fund.models.dysmorphology_studies import bench_dysmorphology_studies
from quant_fund.models.genetic_diagnostics import bench_genetic_diagnostics
from quant_fund.models.lysosomal_medicine import bench_lysosomal_medicine
from quant_fund.models.medical_genetics_studies import bench_medical_genetics_studies
from quant_fund.models.mitochondrial_medicine import bench_mitochondrial_medicine
from quant_fund.models.pharmacogenomics_studies import bench_pharmacogenomics_studies


def test_medical_genetics_studies():
    assert bench_medical_genetics_studies()["synthetic_medical_genetics_studies"] == 1.0


def test_genetic_diagnostics():
    assert bench_genetic_diagnostics()["synthetic_genetic_diagnostics"] == 1.0


def test_lysosomal_medicine():
    assert bench_lysosomal_medicine()["synthetic_lysosomal_medicine"] == 1.0


def test_mitochondrial_medicine():
    assert bench_mitochondrial_medicine()["synthetic_mitochondrial_medicine"] == 1.0


def test_dysmorphology_studies():
    assert bench_dysmorphology_studies()["synthetic_dysmorphology_studies"] == 1.0


def test_pharmacogenomics_studies():
    assert bench_pharmacogenomics_studies()["synthetic_pharmacogenomics_studies"] == 1.0
