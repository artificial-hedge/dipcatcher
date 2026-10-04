"""Wave-1065 geography canon tests."""

from __future__ import annotations

from quant_fund.models.cartography import bench_cartography
from quant_fund.models.climatology import bench_climatology
from quant_fund.models.geomorphology import bench_geomorphology
from quant_fund.models.human_geography import bench_human_geography
from quant_fund.models.physical_geography import bench_physical_geography
from quant_fund.models.remote_sensing import bench_remote_sensing


def test_physical_geography():
    assert bench_physical_geography()["synthetic_physical_geography"] == 1.0


def test_human_geography():
    assert bench_human_geography()["synthetic_human_geography"] == 1.0


def test_cartography():
    assert bench_cartography()["synthetic_cartography"] == 1.0


def test_remote_sensing():
    assert bench_remote_sensing()["synthetic_remote_sensing"] == 1.0


def test_geomorphology():
    assert bench_geomorphology()["synthetic_geomorphology"] == 1.0


def test_climatology():
    assert bench_climatology()["synthetic_climatology"] == 1.0
