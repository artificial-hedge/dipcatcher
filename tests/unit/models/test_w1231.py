"""Wave-1231 transplant/immunology canon tests."""

from __future__ import annotations

from quant_fund.models.allergy_studies import bench_allergy_studies
from quant_fund.models.autoimmunity_studies import bench_autoimmunity_studies
from quant_fund.models.hematopoietic_transplant import bench_hematopoietic_transplant
from quant_fund.models.immunodeficiency_studies import bench_immunodeficiency_studies
from quant_fund.models.immunology_medicine import bench_immunology_medicine
from quant_fund.models.transplant_medicine_studies import bench_transplant_medicine_studies


def test_transplant_medicine_studies():
    assert bench_transplant_medicine_studies()["synthetic_transplant_medicine_studies"] == 1.0


def test_immunology_medicine():
    assert bench_immunology_medicine()["synthetic_immunology_medicine"] == 1.0


def test_allergy_studies():
    assert bench_allergy_studies()["synthetic_allergy_studies"] == 1.0


def test_autoimmunity_studies():
    assert bench_autoimmunity_studies()["synthetic_autoimmunity_studies"] == 1.0


def test_hematopoietic_transplant():
    assert bench_hematopoietic_transplant()["synthetic_hematopoietic_transplant"] == 1.0


def test_immunodeficiency_studies():
    assert bench_immunodeficiency_studies()["synthetic_immunodeficiency_studies"] == 1.0
