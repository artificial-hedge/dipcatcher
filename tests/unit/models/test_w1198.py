"""Wave-1198 procedural-medicine canon tests."""

from __future__ import annotations

from quant_fund.models.electrodiagnostic_studies import bench_electrodiagnostic_studies
from quant_fund.models.hyperbaric_medicine import bench_hyperbaric_medicine
from quant_fund.models.infusion_therapy import bench_infusion_therapy
from quant_fund.models.pain_management import bench_pain_management
from quant_fund.models.sleep_medicine import bench_sleep_medicine
from quant_fund.models.wound_care import bench_wound_care


def test_sleep_medicine():
    assert bench_sleep_medicine()["synthetic_sleep_medicine"] == 1.0


def test_pain_management():
    assert bench_pain_management()["synthetic_pain_management"] == 1.0


def test_wound_care():
    assert bench_wound_care()["synthetic_wound_care"] == 1.0


def test_infusion_therapy():
    assert bench_infusion_therapy()["synthetic_infusion_therapy"] == 1.0


def test_hyperbaric_medicine():
    assert bench_hyperbaric_medicine()["synthetic_hyperbaric_medicine"] == 1.0


def test_electrodiagnostic_studies():
    assert bench_electrodiagnostic_studies()["synthetic_electrodiagnostic_studies"] == 1.0
