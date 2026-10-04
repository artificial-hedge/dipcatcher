"""Wave-1220 infectious-immune canon tests."""

from __future__ import annotations

from quant_fund.models.allergy_immunology import bench_allergy_immunology
from quant_fund.models.antimicrobial_stewardship import bench_antimicrobial_stewardship
from quant_fund.models.hiv_medicine import bench_hiv_medicine
from quant_fund.models.immunology_studies import bench_immunology_studies
from quant_fund.models.infectious_disease_medicine import bench_infectious_disease_medicine
from quant_fund.models.rheumatology_studies import bench_rheumatology_studies


def test_infectious_disease_medicine():
    assert bench_infectious_disease_medicine()["synthetic_infectious_disease_medicine"] == 1.0


def test_hiv_medicine():
    assert bench_hiv_medicine()["synthetic_hiv_medicine"] == 1.0


def test_antimicrobial_stewardship():
    assert bench_antimicrobial_stewardship()["synthetic_antimicrobial_stewardship"] == 1.0


def test_rheumatology_studies():
    assert bench_rheumatology_studies()["synthetic_rheumatology_studies"] == 1.0


def test_immunology_studies():
    assert bench_immunology_studies()["synthetic_immunology_studies"] == 1.0


def test_allergy_immunology():
    assert bench_allergy_immunology()["synthetic_allergy_immunology"] == 1.0
