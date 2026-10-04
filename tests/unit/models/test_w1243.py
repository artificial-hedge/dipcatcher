"""Wave-1243 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.healthcare_infection_studies import bench_healthcare_infection_studies
from quant_fund.models.mycosis_studies import bench_mycosis_studies
from quant_fund.models.opportunistic_studies import bench_opportunistic_studies
from quant_fund.models.sepsis_studies import bench_sepsis_studies
from quant_fund.models.sexually_transmitted_studies import bench_sexually_transmitted_studies
from quant_fund.models.tuberculosis_studies import bench_tuberculosis_studies


def test_sepsis_studies() -> None:
    assert bench_sepsis_studies()["synthetic_sepsis_studies"] == 1.0


def test_tuberculosis_studies() -> None:
    assert bench_tuberculosis_studies()["synthetic_tuberculosis_studies"] == 1.0


def test_mycosis_studies() -> None:
    assert bench_mycosis_studies()["synthetic_mycosis_studies"] == 1.0


def test_sexually_transmitted_studies() -> None:
    assert bench_sexually_transmitted_studies()["synthetic_sexually_transmitted_studies"] == 1.0


def test_healthcare_infection_studies() -> None:
    assert bench_healthcare_infection_studies()["synthetic_healthcare_infection_studies"] == 1.0


def test_opportunistic_studies() -> None:
    assert bench_opportunistic_studies()["synthetic_opportunistic_studies"] == 1.0
