"""Wave-1156 mathematical-sciences canon tests."""

from __future__ import annotations

from quant_fund.models.applied_mathematics import bench_applied_mathematics
from quant_fund.models.bioinformatics_5 import bench_bioinformatics_5
from quant_fund.models.computational_science import bench_computational_science
from quant_fund.models.data_science import bench_data_science
from quant_fund.models.probability_4 import bench_probability_4
from quant_fund.models.statistics_2 import bench_statistics_2


def test_applied_mathematics():
    assert bench_applied_mathematics()["synthetic_applied_mathematics"] == 1.0


def test_statistics_2():
    assert bench_statistics_2()["synthetic_statistics_2"] == 1.0


def test_probability_4():
    assert bench_probability_4()["synthetic_probability_4"] == 1.0


def test_computational_science():
    assert bench_computational_science()["synthetic_computational_science"] == 1.0


def test_data_science():
    assert bench_data_science()["synthetic_data_science"] == 1.0


def test_bioinformatics_5():
    assert bench_bioinformatics_5()["synthetic_bioinformatics_5"] == 1.0
