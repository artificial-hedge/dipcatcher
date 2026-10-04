"""Wave-1070 sports science canon tests."""

from __future__ import annotations

from quant_fund.models.athletic_training import bench_athletic_training
from quant_fund.models.exercise_physiology import bench_exercise_physiology
from quant_fund.models.sports_analytics import bench_sports_analytics
from quant_fund.models.sports_biomechanics import bench_sports_biomechanics
from quant_fund.models.sports_psychology import bench_sports_psychology
from quant_fund.models.sports_science import bench_sports_science


def test_sports_science():
    assert bench_sports_science()["synthetic_sports_science"] == 1.0


def test_exercise_physiology():
    assert bench_exercise_physiology()["synthetic_exercise_physiology"] == 1.0


def test_sports_biomechanics():
    assert bench_sports_biomechanics()["synthetic_sports_biomechanics"] == 1.0


def test_sports_psychology():
    assert bench_sports_psychology()["synthetic_sports_psychology"] == 1.0


def test_athletic_training():
    assert bench_athletic_training()["synthetic_athletic_training"] == 1.0


def test_sports_analytics():
    assert bench_sports_analytics()["synthetic_sports_analytics"] == 1.0
