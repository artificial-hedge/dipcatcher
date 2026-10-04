"""Wave-1189 hospitality canon tests."""

from __future__ import annotations

from quant_fund.models.event_management import bench_event_management
from quant_fund.models.hospitality_studies import bench_hospitality_studies
from quant_fund.models.hotel_management import bench_hotel_management
from quant_fund.models.leisure_science import bench_leisure_science
from quant_fund.models.recreation_management import bench_recreation_management
from quant_fund.models.tourism_studies import bench_tourism_studies


def test_hospitality_studies():
    assert bench_hospitality_studies()["synthetic_hospitality_studies"] == 1.0


def test_event_management():
    assert bench_event_management()["synthetic_event_management"] == 1.0


def test_hotel_management():
    assert bench_hotel_management()["synthetic_hotel_management"] == 1.0


def test_tourism_studies():
    assert bench_tourism_studies()["synthetic_tourism_studies"] == 1.0


def test_recreation_management():
    assert bench_recreation_management()["synthetic_recreation_management"] == 1.0


def test_leisure_science():
    assert bench_leisure_science()["synthetic_leisure_science"] == 1.0
