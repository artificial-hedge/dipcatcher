"""Wave-1218 nephrology canon tests."""

from __future__ import annotations

from quant_fund.models.acid_base_medicine import bench_acid_base_medicine
from quant_fund.models.dialysis_medicine import bench_dialysis_medicine
from quant_fund.models.hypertension_medicine import bench_hypertension_medicine
from quant_fund.models.nephrology_studies import bench_nephrology_studies
from quant_fund.models.renal_transplant import bench_renal_transplant
from quant_fund.models.urology_studies import bench_urology_studies


def test_nephrology_studies():
    assert bench_nephrology_studies()["synthetic_nephrology_studies"] == 1.0


def test_dialysis_medicine():
    assert bench_dialysis_medicine()["synthetic_dialysis_medicine"] == 1.0


def test_renal_transplant():
    assert bench_renal_transplant()["synthetic_renal_transplant"] == 1.0


def test_acid_base_medicine():
    assert bench_acid_base_medicine()["synthetic_acid_base_medicine"] == 1.0


def test_hypertension_medicine():
    assert bench_hypertension_medicine()["synthetic_hypertension_medicine"] == 1.0


def test_urology_studies():
    assert bench_urology_studies()["synthetic_urology_studies"] == 1.0
