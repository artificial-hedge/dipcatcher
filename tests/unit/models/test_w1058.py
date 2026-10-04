"""Wave-1058 philosophy canon tests."""

from __future__ import annotations

from quant_fund.models.aesthetics import bench_aesthetics
from quant_fund.models.epistemology import bench_epistemology
from quant_fund.models.ethics_philosophy import bench_ethics_philosophy
from quant_fund.models.logic_philosophy import bench_logic_philosophy
from quant_fund.models.metaphysics import bench_metaphysics
from quant_fund.models.philosophy_of_science import bench_philosophy_of_science


def test_metaphysics():
    assert bench_metaphysics()["synthetic_metaphysics"] == 1.0


def test_epistemology():
    assert bench_epistemology()["synthetic_epistemology"] == 1.0


def test_ethics_philosophy():
    assert bench_ethics_philosophy()["synthetic_ethics_philosophy"] == 1.0


def test_logic_philosophy():
    assert bench_logic_philosophy()["synthetic_logic_philosophy"] == 1.0


def test_philosophy_of_science():
    assert bench_philosophy_of_science()["synthetic_philosophy_of_science"] == 1.0


def test_aesthetics():
    assert bench_aesthetics()["synthetic_aesthetics"] == 1.0
