"""Wave-1176 theology canon tests."""

from __future__ import annotations

from quant_fund.models.biblical_studies_2 import bench_biblical_studies_2
from quant_fund.models.buddhist_studies_2 import bench_buddhist_studies_2
from quant_fund.models.comparative_religion_2 import bench_comparative_religion_2
from quant_fund.models.islamic_studies_2 import bench_islamic_studies_2
from quant_fund.models.religious_studies_3 import bench_religious_studies_3
from quant_fund.models.theology_3 import bench_theology_3


def test_theology_3():
    assert bench_theology_3()["synthetic_theology_3"] == 1.0


def test_religious_studies_3():
    assert bench_religious_studies_3()["synthetic_religious_studies_3"] == 1.0


def test_comparative_religion_2():
    assert bench_comparative_religion_2()["synthetic_comparative_religion_2"] == 1.0


def test_biblical_studies_2():
    assert bench_biblical_studies_2()["synthetic_biblical_studies_2"] == 1.0


def test_islamic_studies_2():
    assert bench_islamic_studies_2()["synthetic_islamic_studies_2"] == 1.0


def test_buddhist_studies_2():
    assert bench_buddhist_studies_2()["synthetic_buddhist_studies_2"] == 1.0
