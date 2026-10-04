"""Wave-1080 medieval studies canon tests."""

from __future__ import annotations

from quant_fund.models.byzantine_studies import bench_byzantine_studies
from quant_fund.models.codicology import bench_codicology
from quant_fund.models.hagiography import bench_hagiography
from quant_fund.models.medieval_studies import bench_medieval_studies
from quant_fund.models.numismatics import bench_numismatics
from quant_fund.models.paleography import bench_paleography


def test_medieval_studies():
    assert bench_medieval_studies()["synthetic_medieval_studies"] == 1.0


def test_paleography():
    assert bench_paleography()["synthetic_paleography"] == 1.0


def test_codicology():
    assert bench_codicology()["synthetic_codicology"] == 1.0


def test_hagiography():
    assert bench_hagiography()["synthetic_hagiography"] == 1.0


def test_byzantine_studies():
    assert bench_byzantine_studies()["synthetic_byzantine_studies"] == 1.0


def test_numismatics():
    assert bench_numismatics()["synthetic_numismatics"] == 1.0
