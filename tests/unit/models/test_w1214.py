"""Wave-1214 cardiology canon tests."""

from __future__ import annotations

from quant_fund.models.cardiology_studies import bench_cardiology_studies
from quant_fund.models.cardiovascular_imaging import bench_cardiovascular_imaging
from quant_fund.models.electrophysiology_studies import bench_electrophysiology_studies
from quant_fund.models.heart_failure_medicine import bench_heart_failure_medicine
from quant_fund.models.interventional_cardiology import bench_interventional_cardiology
from quant_fund.models.preventive_cardiology import bench_preventive_cardiology


def test_cardiology_studies():
    assert bench_cardiology_studies()["synthetic_cardiology_studies"] == 1.0


def test_interventional_cardiology():
    assert bench_interventional_cardiology()["synthetic_interventional_cardiology"] == 1.0


def test_electrophysiology_studies():
    assert bench_electrophysiology_studies()["synthetic_electrophysiology_studies"] == 1.0


def test_heart_failure_medicine():
    assert bench_heart_failure_medicine()["synthetic_heart_failure_medicine"] == 1.0


def test_preventive_cardiology():
    assert bench_preventive_cardiology()["synthetic_preventive_cardiology"] == 1.0


def test_cardiovascular_imaging():
    assert bench_cardiovascular_imaging()["synthetic_cardiovascular_imaging"] == 1.0
