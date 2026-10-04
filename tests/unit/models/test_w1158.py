"""Wave-1158 social-sciences canon tests."""

from __future__ import annotations

from quant_fund.models.anthropology_6 import bench_anthropology_6
from quant_fund.models.economics_6 import bench_economics_6
from quant_fund.models.linguistics_7 import bench_linguistics_7
from quant_fund.models.political_science_3 import bench_political_science_3
from quant_fund.models.psychology_5 import bench_psychology_5
from quant_fund.models.sociology_6 import bench_sociology_6


def test_sociology_6():
    assert bench_sociology_6()["synthetic_sociology_6"] == 1.0


def test_economics_6():
    assert bench_economics_6()["synthetic_economics_6"] == 1.0


def test_political_science_3():
    assert bench_political_science_3()["synthetic_political_science_3"] == 1.0


def test_psychology_5():
    assert bench_psychology_5()["synthetic_psychology_5"] == 1.0


def test_anthropology_6():
    assert bench_anthropology_6()["synthetic_anthropology_6"] == 1.0


def test_linguistics_7():
    assert bench_linguistics_7()["synthetic_linguistics_7"] == 1.0
