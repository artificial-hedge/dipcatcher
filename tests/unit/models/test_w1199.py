"""Wave-1199 interventional-medicine canon tests."""

from __future__ import annotations

from quant_fund.models.cardiac_electrophysiology import bench_cardiac_electrophysiology
from quant_fund.models.dialysis_technology import bench_dialysis_technology
from quant_fund.models.hepatobiliary_studies import bench_hepatobiliary_studies
from quant_fund.models.interventional_radiology import bench_interventional_radiology
from quant_fund.models.nuclear_cardiology import bench_nuclear_cardiology
from quant_fund.models.transplant_studies import bench_transplant_studies


def test_dialysis_technology():
    assert bench_dialysis_technology()["synthetic_dialysis_technology"] == 1.0


def test_transplant_studies():
    assert bench_transplant_studies()["synthetic_transplant_studies"] == 1.0


def test_hepatobiliary_studies():
    assert bench_hepatobiliary_studies()["synthetic_hepatobiliary_studies"] == 1.0


def test_cardiac_electrophysiology():
    assert bench_cardiac_electrophysiology()["synthetic_cardiac_electrophysiology"] == 1.0


def test_interventional_radiology():
    assert bench_interventional_radiology()["synthetic_interventional_radiology"] == 1.0


def test_nuclear_cardiology():
    assert bench_nuclear_cardiology()["synthetic_nuclear_cardiology"] == 1.0
