"""Wave-1179 security canon tests."""

from __future__ import annotations

from quant_fund.models.conflict_studies import bench_conflict_studies
from quant_fund.models.intelligence_analysis import bench_intelligence_analysis
from quant_fund.models.military_history_2 import bench_military_history_2
from quant_fund.models.peace_research import bench_peace_research
from quant_fund.models.strategic_analysis import bench_strategic_analysis
from quant_fund.models.war_studies import bench_war_studies


def test_war_studies():
    assert bench_war_studies()["synthetic_war_studies"] == 1.0


def test_strategic_analysis():
    assert bench_strategic_analysis()["synthetic_strategic_analysis"] == 1.0


def test_intelligence_analysis():
    assert bench_intelligence_analysis()["synthetic_intelligence_analysis"] == 1.0


def test_peace_research():
    assert bench_peace_research()["synthetic_peace_research"] == 1.0


def test_conflict_studies():
    assert bench_conflict_studies()["synthetic_conflict_studies"] == 1.0


def test_military_history_2():
    assert bench_military_history_2()["synthetic_military_history_2"] == 1.0
