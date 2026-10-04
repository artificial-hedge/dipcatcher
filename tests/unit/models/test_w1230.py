"""Wave-1230 emergency-medicine canon tests."""

from __future__ import annotations

from quant_fund.models.acute_care_studies import bench_acute_care_studies
from quant_fund.models.disaster_medicine import bench_disaster_medicine
from quant_fund.models.emergency_medicine_studies import bench_emergency_medicine_studies
from quant_fund.models.resuscitation_medicine import bench_resuscitation_medicine
from quant_fund.models.toxicology_medicine import bench_toxicology_medicine
from quant_fund.models.trauma_medicine import bench_trauma_medicine


def test_emergency_medicine_studies():
    assert bench_emergency_medicine_studies()["synthetic_emergency_medicine_studies"] == 1.0


def test_trauma_medicine():
    assert bench_trauma_medicine()["synthetic_trauma_medicine"] == 1.0


def test_toxicology_medicine():
    assert bench_toxicology_medicine()["synthetic_toxicology_medicine"] == 1.0


def test_disaster_medicine():
    assert bench_disaster_medicine()["synthetic_disaster_medicine"] == 1.0


def test_acute_care_studies():
    assert bench_acute_care_studies()["synthetic_acute_care_studies"] == 1.0


def test_resuscitation_medicine():
    assert bench_resuscitation_medicine()["synthetic_resuscitation_medicine"] == 1.0
