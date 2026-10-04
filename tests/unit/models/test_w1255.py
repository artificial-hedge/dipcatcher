"""Wave-1255 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.als_studies import bench_als_studies
from quant_fund.models.alzheimer_studies import bench_alzheimer_studies
from quant_fund.models.dementia_studies import bench_dementia_studies
from quant_fund.models.huntington_studies import bench_huntington_studies
from quant_fund.models.ms_studies import bench_ms_studies
from quant_fund.models.parkinson_studies import bench_parkinson_studies


def test_parkinson_studies() -> None:
    assert bench_parkinson_studies()["synthetic_parkinson_studies"] == 1.0


def test_alzheimer_studies() -> None:
    assert bench_alzheimer_studies()["synthetic_alzheimer_studies"] == 1.0


def test_ms_studies() -> None:
    assert bench_ms_studies()["synthetic_ms_studies"] == 1.0


def test_als_studies() -> None:
    assert bench_als_studies()["synthetic_als_studies"] == 1.0


def test_huntington_studies() -> None:
    assert bench_huntington_studies()["synthetic_huntington_studies"] == 1.0


def test_dementia_studies() -> None:
    assert bench_dementia_studies()["synthetic_dementia_studies"] == 1.0
