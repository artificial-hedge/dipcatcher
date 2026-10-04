"""Wave-1232 geriatrics canon tests."""

from __future__ import annotations

from quant_fund.models.caregiver_medicine import bench_caregiver_medicine
from quant_fund.models.falls_prevention_studies import bench_falls_prevention_studies
from quant_fund.models.frailty_medicine import bench_frailty_medicine
from quant_fund.models.geriatrics_studies import bench_geriatrics_studies
from quant_fund.models.memory_clinic_studies import bench_memory_clinic_studies
from quant_fund.models.polypharmacy_studies import bench_polypharmacy_studies


def test_geriatrics_studies():
    assert bench_geriatrics_studies()["synthetic_geriatrics_studies"] == 1.0


def test_frailty_medicine():
    assert bench_frailty_medicine()["synthetic_frailty_medicine"] == 1.0


def test_memory_clinic_studies():
    assert bench_memory_clinic_studies()["synthetic_memory_clinic_studies"] == 1.0


def test_falls_prevention_studies():
    assert bench_falls_prevention_studies()["synthetic_falls_prevention_studies"] == 1.0


def test_polypharmacy_studies():
    assert bench_polypharmacy_studies()["synthetic_polypharmacy_studies"] == 1.0


def test_caregiver_medicine():
    assert bench_caregiver_medicine()["synthetic_caregiver_medicine"] == 1.0
