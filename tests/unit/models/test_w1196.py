"""Wave-1196 emergency-safety canon tests."""

from __future__ import annotations

from quant_fund.models.disaster_management import bench_disaster_management
from quant_fund.models.emergency_medical_technician import bench_emergency_medical_technician
from quant_fund.models.fire_science_studies import bench_fire_science_studies
from quant_fund.models.industrial_hygiene import bench_industrial_hygiene
from quant_fund.models.occupational_safety import bench_occupational_safety
from quant_fund.models.paramedic_studies import bench_paramedic_studies


def test_emergency_medical_technician():
    assert bench_emergency_medical_technician()["synthetic_emergency_medical_technician"] == 1.0


def test_fire_science_studies():
    assert bench_fire_science_studies()["synthetic_fire_science_studies"] == 1.0


def test_paramedic_studies():
    assert bench_paramedic_studies()["synthetic_paramedic_studies"] == 1.0


def test_disaster_management():
    assert bench_disaster_management()["synthetic_disaster_management"] == 1.0


def test_occupational_safety():
    assert bench_occupational_safety()["synthetic_occupational_safety"] == 1.0


def test_industrial_hygiene():
    assert bench_industrial_hygiene()["synthetic_industrial_hygiene"] == 1.0
