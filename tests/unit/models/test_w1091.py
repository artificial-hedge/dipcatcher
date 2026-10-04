"""Wave-1091 education-2 canon tests."""

from __future__ import annotations

from quant_fund.models.comparative_education import bench_comparative_education
from quant_fund.models.distance_learning import bench_distance_learning
from quant_fund.models.higher_education import bench_higher_education
from quant_fund.models.literacy_studies import bench_literacy_studies
from quant_fund.models.special_education import bench_special_education
from quant_fund.models.vocational_education import bench_vocational_education


def test_higher_education():
    assert bench_higher_education()["synthetic_higher_education"] == 1.0


def test_vocational_education():
    assert bench_vocational_education()["synthetic_vocational_education"] == 1.0


def test_special_education():
    assert bench_special_education()["synthetic_special_education"] == 1.0


def test_comparative_education():
    assert bench_comparative_education()["synthetic_comparative_education"] == 1.0


def test_literacy_studies():
    assert bench_literacy_studies()["synthetic_literacy_studies"] == 1.0


def test_distance_learning():
    assert bench_distance_learning()["synthetic_distance_learning"] == 1.0
