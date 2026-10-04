"""Wave-1116 psychology-3 canon tests."""

from __future__ import annotations

from quant_fund.models.comparative_psychology import bench_comparative_psychology
from quant_fund.models.environmental_psychology import bench_environmental_psychology
from quant_fund.models.evolutionary_psychology import bench_evolutionary_psychology
from quant_fund.models.experimental_psychology import bench_experimental_psychology
from quant_fund.models.psychopathology import bench_psychopathology
from quant_fund.models.sport_psychology import bench_sport_psychology


def test_experimental_psychology():
    assert bench_experimental_psychology()["synthetic_experimental_psychology"] == 1.0


def test_comparative_psychology():
    assert bench_comparative_psychology()["synthetic_comparative_psychology"] == 1.0


def test_evolutionary_psychology():
    assert bench_evolutionary_psychology()["synthetic_evolutionary_psychology"] == 1.0


def test_psychopathology():
    assert bench_psychopathology()["synthetic_psychopathology"] == 1.0


def test_environmental_psychology():
    assert bench_environmental_psychology()["synthetic_environmental_psychology"] == 1.0


def test_sport_psychology():
    assert bench_sport_psychology()["synthetic_sport_psychology"] == 1.0
