"""Wave-1246 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.bariatric_surgery_studies import bench_bariatric_surgery_studies
from quant_fund.models.burn_surgery_studies import bench_burn_surgery_studies
from quant_fund.models.endocrine_surgery_studies import bench_endocrine_surgery_studies
from quant_fund.models.pediatric_surgery_studies import bench_pediatric_surgery_studies
from quant_fund.models.plastic_surgery_studies import bench_plastic_surgery_studies
from quant_fund.models.transplant_surgery_studies import bench_transplant_surgery_studies


def test_bariatric_surgery_studies() -> None:
    assert bench_bariatric_surgery_studies()["synthetic_bariatric_surgery_studies"] == 1.0


def test_pediatric_surgery_studies() -> None:
    assert bench_pediatric_surgery_studies()["synthetic_pediatric_surgery_studies"] == 1.0


def test_plastic_surgery_studies() -> None:
    assert bench_plastic_surgery_studies()["synthetic_plastic_surgery_studies"] == 1.0


def test_burn_surgery_studies() -> None:
    assert bench_burn_surgery_studies()["synthetic_burn_surgery_studies"] == 1.0


def test_endocrine_surgery_studies() -> None:
    assert bench_endocrine_surgery_studies()["synthetic_endocrine_surgery_studies"] == 1.0


def test_transplant_surgery_studies() -> None:
    assert bench_transplant_surgery_studies()["synthetic_transplant_surgery_studies"] == 1.0
