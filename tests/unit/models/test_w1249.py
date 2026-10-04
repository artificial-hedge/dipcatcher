"""Wave-1249 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.acne_studies import bench_acne_studies
from quant_fund.models.alopecia_studies import bench_alopecia_studies
from quant_fund.models.eczema_studies import bench_eczema_studies
from quant_fund.models.psoriasis_studies import bench_psoriasis_studies
from quant_fund.models.skin_cancer_studies import bench_skin_cancer_studies
from quant_fund.models.vitiligo_studies import bench_vitiligo_studies


def test_skin_cancer_studies() -> None:
    assert bench_skin_cancer_studies()["synthetic_skin_cancer_studies"] == 1.0


def test_psoriasis_studies() -> None:
    assert bench_psoriasis_studies()["synthetic_psoriasis_studies"] == 1.0


def test_eczema_studies() -> None:
    assert bench_eczema_studies()["synthetic_eczema_studies"] == 1.0


def test_acne_studies() -> None:
    assert bench_acne_studies()["synthetic_acne_studies"] == 1.0


def test_vitiligo_studies() -> None:
    assert bench_vitiligo_studies()["synthetic_vitiligo_studies"] == 1.0


def test_alopecia_studies() -> None:
    assert bench_alopecia_studies()["synthetic_alopecia_studies"] == 1.0
