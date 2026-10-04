"""Wave-1068 military/defense studies canon tests."""

from __future__ import annotations

from quant_fund.models.conflict_resolution import bench_conflict_resolution
from quant_fund.models.defense_studies import bench_defense_studies
from quant_fund.models.intelligence_studies import bench_intelligence_studies
from quant_fund.models.military_science import bench_military_science
from quant_fund.models.peace_studies import bench_peace_studies
from quant_fund.models.strategic_studies import bench_strategic_studies


def test_military_science():
    assert bench_military_science()["synthetic_military_science"] == 1.0


def test_defense_studies():
    assert bench_defense_studies()["synthetic_defense_studies"] == 1.0


def test_strategic_studies():
    assert bench_strategic_studies()["synthetic_strategic_studies"] == 1.0


def test_intelligence_studies():
    assert bench_intelligence_studies()["synthetic_intelligence_studies"] == 1.0


def test_peace_studies():
    assert bench_peace_studies()["synthetic_peace_studies"] == 1.0


def test_conflict_resolution():
    assert bench_conflict_resolution()["synthetic_conflict_resolution"] == 1.0
