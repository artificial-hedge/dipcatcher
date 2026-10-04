"""Wave-1221 derm-eye-ent canon tests."""

from __future__ import annotations

from quant_fund.models.audiology_medicine import bench_audiology_medicine
from quant_fund.models.dermatology_studies import bench_dermatology_studies
from quant_fund.models.dermatopathology import bench_dermatopathology
from quant_fund.models.ophthalmology_studies import bench_ophthalmology_studies
from quant_fund.models.optometry_studies import bench_optometry_studies
from quant_fund.models.otolaryngology_studies import bench_otolaryngology_studies


def test_dermatology_studies():
    assert bench_dermatology_studies()["synthetic_dermatology_studies"] == 1.0


def test_ophthalmology_studies():
    assert bench_ophthalmology_studies()["synthetic_ophthalmology_studies"] == 1.0


def test_otolaryngology_studies():
    assert bench_otolaryngology_studies()["synthetic_otolaryngology_studies"] == 1.0


def test_audiology_medicine():
    assert bench_audiology_medicine()["synthetic_audiology_medicine"] == 1.0


def test_optometry_studies():
    assert bench_optometry_studies()["synthetic_optometry_studies"] == 1.0


def test_dermatopathology():
    assert bench_dermatopathology()["synthetic_dermatopathology"] == 1.0
