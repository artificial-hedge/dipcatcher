"""Wave-1086 near-eastern/ancient canon tests."""

from __future__ import annotations

from quant_fund.models.assyriology import bench_assyriology
from quant_fund.models.egyptology import bench_egyptology
from quant_fund.models.indology import bench_indology
from quant_fund.models.iranian_studies import bench_iranian_studies
from quant_fund.models.ottoman_studies import bench_ottoman_studies
from quant_fund.models.sinology import bench_sinology


def test_assyriology():
    assert bench_assyriology()["synthetic_assyriology"] == 1.0


def test_egyptology():
    assert bench_egyptology()["synthetic_egyptology"] == 1.0


def test_sinology():
    assert bench_sinology()["synthetic_sinology"] == 1.0


def test_indology():
    assert bench_indology()["synthetic_indology"] == 1.0


def test_iranian_studies():
    assert bench_iranian_studies()["synthetic_iranian_studies"] == 1.0


def test_ottoman_studies():
    assert bench_ottoman_studies()["synthetic_ottoman_studies"] == 1.0
