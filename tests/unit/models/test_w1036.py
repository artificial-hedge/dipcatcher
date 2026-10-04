"""Wave-1036 petroleum-engineering canon tests."""

from __future__ import annotations

from quant_fund.models.drilling_engineering import bench_drilling_engineering
from quant_fund.models.enhanced_recovery import bench_enhanced_recovery
from quant_fund.models.formation_evaluation import bench_formation_evaluation
from quant_fund.models.production_engineering import bench_production_engineering
from quant_fund.models.reservoir_engineering import bench_reservoir_engineering
from quant_fund.models.well_testing import bench_well_testing


def test_reservoir_engineering():
    assert bench_reservoir_engineering()["synthetic_reservoir_engineering"] == 1.0


def test_drilling_engineering():
    assert bench_drilling_engineering()["synthetic_drilling_engineering"] == 1.0


def test_production_engineering():
    assert bench_production_engineering()["synthetic_production_engineering"] == 1.0


def test_formation_evaluation():
    assert bench_formation_evaluation()["synthetic_formation_evaluation"] == 1.0


def test_well_testing():
    assert bench_well_testing()["synthetic_well_testing"] == 1.0


def test_enhanced_recovery():
    assert bench_enhanced_recovery()["synthetic_enhanced_recovery"] == 1.0
