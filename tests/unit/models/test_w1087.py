"""Wave-1087 documentary-sciences canon tests."""

from __future__ import annotations

from quant_fund.models.diplomatics import bench_diplomatics
from quant_fund.models.epigraphy import bench_epigraphy
from quant_fund.models.genealogy_studies import bench_genealogy_studies
from quant_fund.models.heraldry import bench_heraldry
from quant_fund.models.onomastics import bench_onomastics
from quant_fund.models.sigillography import bench_sigillography


def test_epigraphy():
    assert bench_epigraphy()["synthetic_epigraphy"] == 1.0


def test_diplomatics():
    assert bench_diplomatics()["synthetic_diplomatics"] == 1.0


def test_sigillography():
    assert bench_sigillography()["synthetic_sigillography"] == 1.0


def test_heraldry():
    assert bench_heraldry()["synthetic_heraldry"] == 1.0


def test_genealogy_studies():
    assert bench_genealogy_studies()["synthetic_genealogy_studies"] == 1.0


def test_onomastics():
    assert bench_onomastics()["synthetic_onomastics"] == 1.0
