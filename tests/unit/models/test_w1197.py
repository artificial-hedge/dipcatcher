"""Wave-1197 counseling-neonatal canon tests."""

from __future__ import annotations

from quant_fund.models.addiction_counseling import bench_addiction_counseling
from quant_fund.models.genetic_screening import bench_genetic_screening
from quant_fund.models.neonatology_studies import bench_neonatology_studies
from quant_fund.models.pediatric_therapeutics import bench_pediatric_therapeutics
from quant_fund.models.prenatal_studies import bench_prenatal_studies
from quant_fund.models.rehabilitation_counseling import bench_rehabilitation_counseling


def test_addiction_counseling():
    assert bench_addiction_counseling()["synthetic_addiction_counseling"] == 1.0


def test_rehabilitation_counseling():
    assert bench_rehabilitation_counseling()["synthetic_rehabilitation_counseling"] == 1.0


def test_genetic_screening():
    assert bench_genetic_screening()["synthetic_genetic_screening"] == 1.0


def test_prenatal_studies():
    assert bench_prenatal_studies()["synthetic_prenatal_studies"] == 1.0


def test_neonatology_studies():
    assert bench_neonatology_studies()["synthetic_neonatology_studies"] == 1.0


def test_pediatric_therapeutics():
    assert bench_pediatric_therapeutics()["synthetic_pediatric_therapeutics"] == 1.0
