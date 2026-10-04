"""Wave-1253 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.antibody_studies import bench_antibody_studies
from quant_fund.models.chemokine_studies import bench_chemokine_studies
from quant_fund.models.complement_studies import bench_complement_studies
from quant_fund.models.cytokine_studies import bench_cytokine_studies
from quant_fund.models.interferon_studies import bench_interferon_studies
from quant_fund.models.lymphocyte_studies import bench_lymphocyte_studies


def test_cytokine_studies() -> None:
    assert bench_cytokine_studies()["synthetic_cytokine_studies"] == 1.0


def test_chemokine_studies() -> None:
    assert bench_chemokine_studies()["synthetic_chemokine_studies"] == 1.0


def test_interferon_studies() -> None:
    assert bench_interferon_studies()["synthetic_interferon_studies"] == 1.0


def test_complement_studies() -> None:
    assert bench_complement_studies()["synthetic_complement_studies"] == 1.0


def test_antibody_studies() -> None:
    assert bench_antibody_studies()["synthetic_antibody_studies"] == 1.0


def test_lymphocyte_studies() -> None:
    assert bench_lymphocyte_studies()["synthetic_lymphocyte_studies"] == 1.0
