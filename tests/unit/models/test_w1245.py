"""Wave-1245 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.breast_oncology_studies import bench_breast_oncology_studies
from quant_fund.models.gi_oncology_studies import bench_gi_oncology_studies
from quant_fund.models.immuno_oncology_studies import bench_immuno_oncology_studies
from quant_fund.models.medical_oncology_studies import bench_medical_oncology_studies
from quant_fund.models.targeted_therapy_studies import bench_targeted_therapy_studies
from quant_fund.models.thoracic_oncology_studies import bench_thoracic_oncology_studies


def test_medical_oncology_studies() -> None:
    assert bench_medical_oncology_studies()["synthetic_medical_oncology_studies"] == 1.0


def test_immuno_oncology_studies() -> None:
    assert bench_immuno_oncology_studies()["synthetic_immuno_oncology_studies"] == 1.0


def test_targeted_therapy_studies() -> None:
    assert bench_targeted_therapy_studies()["synthetic_targeted_therapy_studies"] == 1.0


def test_breast_oncology_studies() -> None:
    assert bench_breast_oncology_studies()["synthetic_breast_oncology_studies"] == 1.0


def test_thoracic_oncology_studies() -> None:
    assert bench_thoracic_oncology_studies()["synthetic_thoracic_oncology_studies"] == 1.0


def test_gi_oncology_studies() -> None:
    assert bench_gi_oncology_studies()["synthetic_gi_oncology_studies"] == 1.0
