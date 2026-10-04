"""Wave-1248 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.cataract_studies import bench_cataract_studies
from quant_fund.models.corneal_studies import bench_corneal_studies
from quant_fund.models.glaucoma_studies import bench_glaucoma_studies
from quant_fund.models.macular_studies import bench_macular_studies
from quant_fund.models.refractive_studies import bench_refractive_studies
from quant_fund.models.retinal_studies import bench_retinal_studies


def test_retinal_studies() -> None:
    assert bench_retinal_studies()["synthetic_retinal_studies"] == 1.0


def test_corneal_studies() -> None:
    assert bench_corneal_studies()["synthetic_corneal_studies"] == 1.0


def test_glaucoma_studies() -> None:
    assert bench_glaucoma_studies()["synthetic_glaucoma_studies"] == 1.0


def test_cataract_studies() -> None:
    assert bench_cataract_studies()["synthetic_cataract_studies"] == 1.0


def test_macular_studies() -> None:
    assert bench_macular_studies()["synthetic_macular_studies"] == 1.0


def test_refractive_studies() -> None:
    assert bench_refractive_studies()["synthetic_refractive_studies"] == 1.0
