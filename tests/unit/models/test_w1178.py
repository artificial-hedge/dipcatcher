"""Wave-1178 interdisciplinary canon tests."""

from __future__ import annotations

from quant_fund.models.cognitive_science_2 import bench_cognitive_science_2
from quant_fund.models.complexity_science import bench_complexity_science
from quant_fund.models.futures_studies import bench_futures_studies
from quant_fund.models.human_computer_interaction import bench_human_computer_interaction
from quant_fund.models.interdisciplinary_studies import bench_interdisciplinary_studies
from quant_fund.models.systems_science import bench_systems_science


def test_interdisciplinary_studies():
    assert bench_interdisciplinary_studies()["synthetic_interdisciplinary_studies"] == 1.0


def test_cognitive_science_2():
    assert bench_cognitive_science_2()["synthetic_cognitive_science_2"] == 1.0


def test_futures_studies():
    assert bench_futures_studies()["synthetic_futures_studies"] == 1.0


def test_complexity_science():
    assert bench_complexity_science()["synthetic_complexity_science"] == 1.0


def test_systems_science():
    assert bench_systems_science()["synthetic_systems_science"] == 1.0


def test_human_computer_interaction():
    assert bench_human_computer_interaction()["synthetic_human_computer_interaction"] == 1.0
