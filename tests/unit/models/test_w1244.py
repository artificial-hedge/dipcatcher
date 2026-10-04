"""Wave-1244 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.ct_imaging_studies import bench_ct_imaging_studies
from quant_fund.models.mammography_studies import bench_mammography_studies
from quant_fund.models.mri_studies import bench_mri_studies
from quant_fund.models.neuroradiology_studies import bench_neuroradiology_studies
from quant_fund.models.pet_imaging_studies import bench_pet_imaging_studies
from quant_fund.models.ultrasound_studies import bench_ultrasound_studies


def test_neuroradiology_studies() -> None:
    assert bench_neuroradiology_studies()["synthetic_neuroradiology_studies"] == 1.0


def test_mammography_studies() -> None:
    assert bench_mammography_studies()["synthetic_mammography_studies"] == 1.0


def test_ultrasound_studies() -> None:
    assert bench_ultrasound_studies()["synthetic_ultrasound_studies"] == 1.0


def test_ct_imaging_studies() -> None:
    assert bench_ct_imaging_studies()["synthetic_ct_imaging_studies"] == 1.0


def test_mri_studies() -> None:
    assert bench_mri_studies()["synthetic_mri_studies"] == 1.0


def test_pet_imaging_studies() -> None:
    assert bench_pet_imaging_studies()["synthetic_pet_imaging_studies"] == 1.0
