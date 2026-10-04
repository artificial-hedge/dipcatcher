"""Wave-1247 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.aki_studies import bench_aki_studies
from quant_fund.models.ckd_studies import bench_ckd_studies
from quant_fund.models.electrolyte_studies import bench_electrolyte_studies
from quant_fund.models.glomerular_studies import bench_glomerular_studies
from quant_fund.models.stones_studies import bench_stones_studies
from quant_fund.models.tubulointerstitial_studies import bench_tubulointerstitial_studies


def test_glomerular_studies() -> None:
    assert bench_glomerular_studies()["synthetic_glomerular_studies"] == 1.0


def test_tubulointerstitial_studies() -> None:
    assert bench_tubulointerstitial_studies()["synthetic_tubulointerstitial_studies"] == 1.0


def test_ckd_studies() -> None:
    assert bench_ckd_studies()["synthetic_ckd_studies"] == 1.0


def test_aki_studies() -> None:
    assert bench_aki_studies()["synthetic_aki_studies"] == 1.0


def test_electrolyte_studies() -> None:
    assert bench_electrolyte_studies()["synthetic_electrolyte_studies"] == 1.0


def test_stones_studies() -> None:
    assert bench_stones_studies()["synthetic_stones_studies"] == 1.0
