"""Wave-1229 pediatrics canon tests."""

from __future__ import annotations

from quant_fund.models.adolescent_medicine_studies import bench_adolescent_medicine_studies
from quant_fund.models.developmental_pediatrics import bench_developmental_pediatrics
from quant_fund.models.neonatal_medicine_studies import bench_neonatal_medicine_studies
from quant_fund.models.pediatric_cardiology import bench_pediatric_cardiology
from quant_fund.models.pediatric_oncology import bench_pediatric_oncology
from quant_fund.models.pediatrics_studies import bench_pediatrics_studies


def test_pediatrics_studies():
    assert bench_pediatrics_studies()["synthetic_pediatrics_studies"] == 1.0


def test_neonatal_medicine_studies():
    assert bench_neonatal_medicine_studies()["synthetic_neonatal_medicine_studies"] == 1.0


def test_pediatric_cardiology():
    assert bench_pediatric_cardiology()["synthetic_pediatric_cardiology"] == 1.0


def test_pediatric_oncology():
    assert bench_pediatric_oncology()["synthetic_pediatric_oncology"] == 1.0


def test_adolescent_medicine_studies():
    assert bench_adolescent_medicine_studies()["synthetic_adolescent_medicine_studies"] == 1.0


def test_developmental_pediatrics():
    assert bench_developmental_pediatrics()["synthetic_developmental_pediatrics"] == 1.0
