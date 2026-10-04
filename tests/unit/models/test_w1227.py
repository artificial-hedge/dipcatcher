"""Wave-1227 radiology canon tests."""

from __future__ import annotations

from quant_fund.models.body_imaging import bench_body_imaging
from quant_fund.models.diagnostic_imaging import bench_diagnostic_imaging
from quant_fund.models.interventional_neuroradiology import bench_interventional_neuroradiology
from quant_fund.models.musculoskeletal_imaging import bench_musculoskeletal_imaging
from quant_fund.models.pediatric_imaging import bench_pediatric_imaging
from quant_fund.models.radiology_studies import bench_radiology_studies


def test_radiology_studies():
    assert bench_radiology_studies()["synthetic_radiology_studies"] == 1.0


def test_diagnostic_imaging():
    assert bench_diagnostic_imaging()["synthetic_diagnostic_imaging"] == 1.0


def test_interventional_neuroradiology():
    assert bench_interventional_neuroradiology()["synthetic_interventional_neuroradiology"] == 1.0


def test_pediatric_imaging():
    assert bench_pediatric_imaging()["synthetic_pediatric_imaging"] == 1.0


def test_musculoskeletal_imaging():
    assert bench_musculoskeletal_imaging()["synthetic_musculoskeletal_imaging"] == 1.0


def test_body_imaging():
    assert bench_body_imaging()["synthetic_body_imaging"] == 1.0
