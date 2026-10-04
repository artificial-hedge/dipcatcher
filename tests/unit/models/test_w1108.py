"""Wave-1108 sociology-3 canon tests."""

from __future__ import annotations

from quant_fund.models.cultural_sociology import bench_cultural_sociology
from quant_fund.models.environmental_sociology import bench_environmental_sociology
from quant_fund.models.industrial_sociology import bench_industrial_sociology
from quant_fund.models.political_sociology import bench_political_sociology
from quant_fund.models.sociology_of_education import bench_sociology_of_education
from quant_fund.models.sociology_of_religion import bench_sociology_of_religion


def test_industrial_sociology():
    assert bench_industrial_sociology()["synthetic_industrial_sociology"] == 1.0


def test_political_sociology():
    assert bench_political_sociology()["synthetic_political_sociology"] == 1.0


def test_sociology_of_education():
    assert bench_sociology_of_education()["synthetic_sociology_of_education"] == 1.0


def test_sociology_of_religion():
    assert bench_sociology_of_religion()["synthetic_sociology_of_religion"] == 1.0


def test_environmental_sociology():
    assert bench_environmental_sociology()["synthetic_environmental_sociology"] == 1.0


def test_cultural_sociology():
    assert bench_cultural_sociology()["synthetic_cultural_sociology"] == 1.0
