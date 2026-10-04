"""Wave-1175 trades canon tests."""

from __future__ import annotations

from quant_fund.models.automotive_technology import bench_automotive_technology
from quant_fund.models.carpentry_trades import bench_carpentry_trades
from quant_fund.models.electrical_trades import bench_electrical_trades
from quant_fund.models.plumbing_hvac import bench_plumbing_hvac
from quant_fund.models.refrigeration_technology import bench_refrigeration_technology
from quant_fund.models.welding_technology import bench_welding_technology


def test_electrical_trades():
    assert bench_electrical_trades()["synthetic_electrical_trades"] == 1.0


def test_plumbing_hvac():
    assert bench_plumbing_hvac()["synthetic_plumbing_hvac"] == 1.0


def test_welding_technology():
    assert bench_welding_technology()["synthetic_welding_technology"] == 1.0


def test_carpentry_trades():
    assert bench_carpentry_trades()["synthetic_carpentry_trades"] == 1.0


def test_automotive_technology():
    assert bench_automotive_technology()["synthetic_automotive_technology"] == 1.0


def test_refrigeration_technology():
    assert bench_refrigeration_technology()["synthetic_refrigeration_technology"] == 1.0
