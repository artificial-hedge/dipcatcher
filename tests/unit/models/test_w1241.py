"""Wave-1241 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.anemia_studies import bench_anemia_studies
from quant_fund.models.bleeding_disorders import bench_bleeding_disorders
from quant_fund.models.coagulation_studies import bench_coagulation_studies
from quant_fund.models.hemoglobin_studies import bench_hemoglobin_studies
from quant_fund.models.marrow_studies import bench_marrow_studies
from quant_fund.models.thrombosis_medicine import bench_thrombosis_medicine


def test_anemia_studies() -> None:
    assert bench_anemia_studies()["synthetic_anemia_studies"] == 1.0


def test_coagulation_studies() -> None:
    assert bench_coagulation_studies()["synthetic_coagulation_studies"] == 1.0


def test_hemoglobin_studies() -> None:
    assert bench_hemoglobin_studies()["synthetic_hemoglobin_studies"] == 1.0


def test_thrombosis_medicine() -> None:
    assert bench_thrombosis_medicine()["synthetic_thrombosis_medicine"] == 1.0


def test_bleeding_disorders() -> None:
    assert bench_bleeding_disorders()["synthetic_bleeding_disorders"] == 1.0


def test_marrow_studies() -> None:
    assert bench_marrow_studies()["synthetic_marrow_studies"] == 1.0
