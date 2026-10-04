"""Wave-1252 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.hypothalamic_studies import bench_hypothalamic_studies
from quant_fund.models.lipid_studies import bench_lipid_studies
from quant_fund.models.metabolic_syndrome_studies import bench_metabolic_syndrome_studies
from quant_fund.models.obesity_studies import bench_obesity_studies
from quant_fund.models.parathyroid_studies import bench_parathyroid_studies
from quant_fund.models.pituitary_studies import bench_pituitary_studies


def test_pituitary_studies() -> None:
    assert bench_pituitary_studies()["synthetic_pituitary_studies"] == 1.0


def test_parathyroid_studies() -> None:
    assert bench_parathyroid_studies()["synthetic_parathyroid_studies"] == 1.0


def test_lipid_studies() -> None:
    assert bench_lipid_studies()["synthetic_lipid_studies"] == 1.0


def test_obesity_studies() -> None:
    assert bench_obesity_studies()["synthetic_obesity_studies"] == 1.0


def test_metabolic_syndrome_studies() -> None:
    assert bench_metabolic_syndrome_studies()["synthetic_metabolic_syndrome_studies"] == 1.0


def test_hypothalamic_studies() -> None:
    assert bench_hypothalamic_studies()["synthetic_hypothalamic_studies"] == 1.0
