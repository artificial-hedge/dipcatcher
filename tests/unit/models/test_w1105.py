"""Wave-1105 behavioral-econ-2 canon tests."""

from __future__ import annotations

from quant_fund.models.bounded_rationality import bench_bounded_rationality
from quant_fund.models.experimental_economics import bench_experimental_economics
from quant_fund.models.financial_behavior import bench_financial_behavior
from quant_fund.models.neuroeconomics import bench_neuroeconomics
from quant_fund.models.nudge_theory import bench_nudge_theory
from quant_fund.models.prospect_theory import bench_prospect_theory


def test_prospect_theory():
    assert bench_prospect_theory()["synthetic_prospect_theory"] == 1.0


def test_bounded_rationality():
    assert bench_bounded_rationality()["synthetic_bounded_rationality"] == 1.0


def test_nudge_theory():
    assert bench_nudge_theory()["synthetic_nudge_theory"] == 1.0


def test_neuroeconomics():
    assert bench_neuroeconomics()["synthetic_neuroeconomics"] == 1.0


def test_experimental_economics():
    assert bench_experimental_economics()["synthetic_experimental_economics"] == 1.0


def test_financial_behavior():
    assert bench_financial_behavior()["synthetic_financial_behavior"] == 1.0
