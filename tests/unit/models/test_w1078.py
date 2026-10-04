"""Wave-1078 performing arts canon tests."""

from __future__ import annotations

from quant_fund.models.choreography import bench_choreography
from quant_fund.models.dance_studies import bench_dance_studies
from quant_fund.models.dramaturgy import bench_dramaturgy
from quant_fund.models.performance_theory import bench_performance_theory
from quant_fund.models.stage_design import bench_stage_design
from quant_fund.models.theater_studies import bench_theater_studies


def test_theater_studies():
    assert bench_theater_studies()["synthetic_theater_studies"] == 1.0


def test_dance_studies():
    assert bench_dance_studies()["synthetic_dance_studies"] == 1.0


def test_performance_theory():
    assert bench_performance_theory()["synthetic_performance_theory"] == 1.0


def test_dramaturgy():
    assert bench_dramaturgy()["synthetic_dramaturgy"] == 1.0


def test_choreography():
    assert bench_choreography()["synthetic_choreography"] == 1.0


def test_stage_design():
    assert bench_stage_design()["synthetic_stage_design"] == 1.0
