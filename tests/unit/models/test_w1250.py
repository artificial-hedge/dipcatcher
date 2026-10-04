"""Wave-1250 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.cochlear_studies import bench_cochlear_studies
from quant_fund.models.head_neck_surgery_studies import bench_head_neck_surgery_studies
from quant_fund.models.laryngology_studies import bench_laryngology_studies
from quant_fund.models.otology_studies import bench_otology_studies
from quant_fund.models.rhinology_studies import bench_rhinology_studies
from quant_fund.models.sinus_studies import bench_sinus_studies


def test_sinus_studies() -> None:
    assert bench_sinus_studies()["synthetic_sinus_studies"] == 1.0


def test_laryngology_studies() -> None:
    assert bench_laryngology_studies()["synthetic_laryngology_studies"] == 1.0


def test_otology_studies() -> None:
    assert bench_otology_studies()["synthetic_otology_studies"] == 1.0


def test_rhinology_studies() -> None:
    assert bench_rhinology_studies()["synthetic_rhinology_studies"] == 1.0


def test_head_neck_surgery_studies() -> None:
    assert bench_head_neck_surgery_studies()["synthetic_head_neck_surgery_studies"] == 1.0


def test_cochlear_studies() -> None:
    assert bench_cochlear_studies()["synthetic_cochlear_studies"] == 1.0
