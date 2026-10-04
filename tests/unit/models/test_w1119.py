"""Wave-1119 history-2 canon tests."""

from __future__ import annotations

from quant_fund.models.cultural_history import bench_cultural_history
from quant_fund.models.diplomatic_history import bench_diplomatic_history
from quant_fund.models.history_of_medicine import bench_history_of_medicine
from quant_fund.models.history_of_technology import bench_history_of_technology
from quant_fund.models.military_history import bench_military_history
from quant_fund.models.social_history import bench_social_history


def test_social_history():
    assert bench_social_history()["synthetic_social_history"] == 1.0


def test_cultural_history():
    assert bench_cultural_history()["synthetic_cultural_history"] == 1.0


def test_military_history():
    assert bench_military_history()["synthetic_military_history"] == 1.0


def test_diplomatic_history():
    assert bench_diplomatic_history()["synthetic_diplomatic_history"] == 1.0


def test_history_of_technology():
    assert bench_history_of_technology()["synthetic_history_of_technology"] == 1.0


def test_history_of_medicine():
    assert bench_history_of_medicine()["synthetic_history_of_medicine"] == 1.0
