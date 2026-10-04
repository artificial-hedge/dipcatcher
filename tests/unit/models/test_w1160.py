"""Wave-1160 humanities canon tests."""

from __future__ import annotations

from quant_fund.models.area_studies_2 import bench_area_studies_2
from quant_fund.models.classics_2 import bench_classics_2
from quant_fund.models.history_5 import bench_history_5
from quant_fund.models.humanities_2 import bench_humanities_2
from quant_fund.models.philosophy_6 import bench_philosophy_6
from quant_fund.models.religious_studies_2 import bench_religious_studies_2


def test_philosophy_6():
    assert bench_philosophy_6()["synthetic_philosophy_6"] == 1.0


def test_history_5():
    assert bench_history_5()["synthetic_history_5"] == 1.0


def test_religious_studies_2():
    assert bench_religious_studies_2()["synthetic_religious_studies_2"] == 1.0


def test_classics_2():
    assert bench_classics_2()["synthetic_classics_2"] == 1.0


def test_area_studies_2():
    assert bench_area_studies_2()["synthetic_area_studies_2"] == 1.0


def test_humanities_2():
    assert bench_humanities_2()["synthetic_humanities_2"] == 1.0
