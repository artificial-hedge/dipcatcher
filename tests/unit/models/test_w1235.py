"""Wave-1235 rheumatology canon tests."""

from __future__ import annotations

from quant_fund.models.connective_tissue_studies import bench_connective_tissue_studies
from quant_fund.models.inflammatory_arthritis_studies import bench_inflammatory_arthritis_studies
from quant_fund.models.myositis_studies import bench_myositis_studies
from quant_fund.models.osteoarthritis_studies import bench_osteoarthritis_studies
from quant_fund.models.rheumatology_medicine import bench_rheumatology_medicine
from quant_fund.models.spondyloarthritis_studies import bench_spondyloarthritis_studies


def test_rheumatology_medicine():
    assert bench_rheumatology_medicine()["synthetic_rheumatology_medicine"] == 1.0


def test_spondyloarthritis_studies():
    assert bench_spondyloarthritis_studies()["synthetic_spondyloarthritis_studies"] == 1.0


def test_inflammatory_arthritis_studies():
    assert bench_inflammatory_arthritis_studies()["synthetic_inflammatory_arthritis_studies"] == 1.0


def test_connective_tissue_studies():
    assert bench_connective_tissue_studies()["synthetic_connective_tissue_studies"] == 1.0


def test_osteoarthritis_studies():
    assert bench_osteoarthritis_studies()["synthetic_osteoarthritis_studies"] == 1.0


def test_myositis_studies():
    assert bench_myositis_studies()["synthetic_myositis_studies"] == 1.0
