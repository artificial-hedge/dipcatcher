"""Wave-1228 dentistry canon tests."""

from __future__ import annotations

from quant_fund.models.dental_studies import bench_dental_studies
from quant_fund.models.endodontic_studies import bench_endodontic_studies
from quant_fund.models.oral_surgery_studies import bench_oral_surgery_studies
from quant_fund.models.orthodontic_studies import bench_orthodontic_studies
from quant_fund.models.pediatric_dentistry import bench_pediatric_dentistry
from quant_fund.models.periodontal_studies import bench_periodontal_studies


def test_dental_studies():
    assert bench_dental_studies()["synthetic_dental_studies"] == 1.0


def test_oral_surgery_studies():
    assert bench_oral_surgery_studies()["synthetic_oral_surgery_studies"] == 1.0


def test_endodontic_studies():
    assert bench_endodontic_studies()["synthetic_endodontic_studies"] == 1.0


def test_periodontal_studies():
    assert bench_periodontal_studies()["synthetic_periodontal_studies"] == 1.0


def test_orthodontic_studies():
    assert bench_orthodontic_studies()["synthetic_orthodontic_studies"] == 1.0


def test_pediatric_dentistry():
    assert bench_pediatric_dentistry()["synthetic_pediatric_dentistry"] == 1.0
