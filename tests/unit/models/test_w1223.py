"""Wave-1223 anesthesia canon tests."""

from __future__ import annotations

from quant_fund.models.airway_management import bench_airway_management
from quant_fund.models.anesthesiology_studies import bench_anesthesiology_studies
from quant_fund.models.pain_medicine_studies import bench_pain_medicine_studies
from quant_fund.models.perioperative_medicine import bench_perioperative_medicine
from quant_fund.models.regional_anesthesia import bench_regional_anesthesia
from quant_fund.models.sedation_medicine import bench_sedation_medicine


def test_anesthesiology_studies():
    assert bench_anesthesiology_studies()["synthetic_anesthesiology_studies"] == 1.0


def test_perioperative_medicine():
    assert bench_perioperative_medicine()["synthetic_perioperative_medicine"] == 1.0


def test_pain_medicine_studies():
    assert bench_pain_medicine_studies()["synthetic_pain_medicine_studies"] == 1.0


def test_regional_anesthesia():
    assert bench_regional_anesthesia()["synthetic_regional_anesthesia"] == 1.0


def test_sedation_medicine():
    assert bench_sedation_medicine()["synthetic_sedation_medicine"] == 1.0


def test_airway_management():
    assert bench_airway_management()["synthetic_airway_management"] == 1.0
