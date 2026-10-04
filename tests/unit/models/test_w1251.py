"""Wave-1251 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.andrology_studies import bench_andrology_studies
from quant_fund.models.bladder_studies import bench_bladder_studies
from quant_fund.models.bph_studies import bench_bph_studies
from quant_fund.models.erectile_studies import bench_erectile_studies
from quant_fund.models.incontinence_studies import bench_incontinence_studies
from quant_fund.models.prostate_studies import bench_prostate_studies


def test_prostate_studies() -> None:
    assert bench_prostate_studies()["synthetic_prostate_studies"] == 1.0


def test_bladder_studies() -> None:
    assert bench_bladder_studies()["synthetic_bladder_studies"] == 1.0


def test_andrology_studies() -> None:
    assert bench_andrology_studies()["synthetic_andrology_studies"] == 1.0


def test_erectile_studies() -> None:
    assert bench_erectile_studies()["synthetic_erectile_studies"] == 1.0


def test_incontinence_studies() -> None:
    assert bench_incontinence_studies()["synthetic_incontinence_studies"] == 1.0


def test_bph_studies() -> None:
    assert bench_bph_studies()["synthetic_bph_studies"] == 1.0
