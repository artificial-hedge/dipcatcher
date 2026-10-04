"""Wave-1174 recreation canon tests."""

from __future__ import annotations

from quant_fund.models.hospitality import bench_hospitality
from quant_fund.models.leisure_studies import bench_leisure_studies
from quant_fund.models.recreation import bench_recreation
from quant_fund.models.recreation_therapy import bench_recreation_therapy
from quant_fund.models.sports_management import bench_sports_management
from quant_fund.models.tourism import bench_tourism


def test_recreation():
    assert bench_recreation()["synthetic_recreation"] == 1.0


def test_leisure_studies():
    assert bench_leisure_studies()["synthetic_leisure_studies"] == 1.0


def test_tourism():
    assert bench_tourism()["synthetic_tourism"] == 1.0


def test_hospitality():
    assert bench_hospitality()["synthetic_hospitality"] == 1.0


def test_sports_management():
    assert bench_sports_management()["synthetic_sports_management"] == 1.0


def test_recreation_therapy():
    assert bench_recreation_therapy()["synthetic_recreation_therapy"] == 1.0
