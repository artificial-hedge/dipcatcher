"""Wave-1200 medical-physics canon tests."""

from __future__ import annotations

from quant_fund.models.cardiovascular_technology import bench_cardiovascular_technology
from quant_fund.models.dosimetry_studies import bench_dosimetry_studies
from quant_fund.models.medical_physics_studies import bench_medical_physics_studies
from quant_fund.models.nuclear_medicine_technology import bench_nuclear_medicine_technology
from quant_fund.models.radiation_dosimetry import bench_radiation_dosimetry
from quant_fund.models.radiopharmacy import bench_radiopharmacy


def test_cardiovascular_technology():
    assert bench_cardiovascular_technology()["synthetic_cardiovascular_technology"] == 1.0


def test_nuclear_medicine_technology():
    assert bench_nuclear_medicine_technology()["synthetic_nuclear_medicine_technology"] == 1.0


def test_radiation_dosimetry():
    assert bench_radiation_dosimetry()["synthetic_radiation_dosimetry"] == 1.0


def test_medical_physics_studies():
    assert bench_medical_physics_studies()["synthetic_medical_physics_studies"] == 1.0


def test_dosimetry_studies():
    assert bench_dosimetry_studies()["synthetic_dosimetry_studies"] == 1.0


def test_radiopharmacy():
    assert bench_radiopharmacy()["synthetic_radiopharmacy"] == 1.0
