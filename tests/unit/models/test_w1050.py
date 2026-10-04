"""Wave-1050 pharmacology canon tests."""

from __future__ import annotations

from quant_fund.models.clinical_pharmacology import bench_clinical_pharmacology
from quant_fund.models.drug_metabolism import bench_drug_metabolism
from quant_fund.models.neuropharmacology import bench_neuropharmacology
from quant_fund.models.pharmacodynamics import bench_pharmacodynamics
from quant_fund.models.pharmacokinetics_2 import bench_pharmacokinetics_2
from quant_fund.models.toxicology import bench_toxicology


def test_pharmacodynamics():
    assert bench_pharmacodynamics()["synthetic_pharmacodynamics"] == 1.0


def test_pharmacokinetics_2():
    assert bench_pharmacokinetics_2()["synthetic_pharmacokinetics_2"] == 1.0


def test_toxicology():
    assert bench_toxicology()["synthetic_toxicology"] == 1.0


def test_clinical_pharmacology():
    assert bench_clinical_pharmacology()["synthetic_clinical_pharmacology"] == 1.0


def test_neuropharmacology():
    assert bench_neuropharmacology()["synthetic_neuropharmacology"] == 1.0


def test_drug_metabolism():
    assert bench_drug_metabolism()["synthetic_drug_metabolism"] == 1.0
