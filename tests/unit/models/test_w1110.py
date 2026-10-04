"""Wave-1110 philosophy-4 canon tests."""

from __future__ import annotations

from quant_fund.models.phenomenology_2 import bench_phenomenology_2
from quant_fund.models.philosophy_of_biology import bench_philosophy_of_biology
from quant_fund.models.philosophy_of_history import bench_philosophy_of_history
from quant_fund.models.philosophy_of_mathematics import bench_philosophy_of_mathematics
from quant_fund.models.philosophy_of_religion import bench_philosophy_of_religion
from quant_fund.models.process_philosophy import bench_process_philosophy


def test_philosophy_of_biology():
    assert bench_philosophy_of_biology()["synthetic_philosophy_of_biology"] == 1.0


def test_philosophy_of_mathematics():
    assert bench_philosophy_of_mathematics()["synthetic_philosophy_of_mathematics"] == 1.0


def test_philosophy_of_religion():
    assert bench_philosophy_of_religion()["synthetic_philosophy_of_religion"] == 1.0


def test_phenomenology_2():
    assert bench_phenomenology_2()["synthetic_phenomenology_2"] == 1.0


def test_philosophy_of_history():
    assert bench_philosophy_of_history()["synthetic_philosophy_of_history"] == 1.0


def test_process_philosophy():
    assert bench_process_philosophy()["synthetic_process_philosophy"] == 1.0
