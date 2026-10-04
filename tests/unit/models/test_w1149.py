"""Wave-1149 clinical-medicine canon tests."""

from __future__ import annotations

from quant_fund.models.dermatology_2 import bench_dermatology_2
from quant_fund.models.hematology_2 import bench_hematology_2
from quant_fund.models.hepatology_2 import bench_hepatology_2
from quant_fund.models.nephrology_2 import bench_nephrology_2
from quant_fund.models.pulmonology_2 import bench_pulmonology_2
from quant_fund.models.toxicology_2 import bench_toxicology_2


def test_toxicology_2():
    assert bench_toxicology_2()["synthetic_toxicology_2"] == 1.0


def test_dermatology_2():
    assert bench_dermatology_2()["synthetic_dermatology_2"] == 1.0


def test_hematology_2():
    assert bench_hematology_2()["synthetic_hematology_2"] == 1.0


def test_pulmonology_2():
    assert bench_pulmonology_2()["synthetic_pulmonology_2"] == 1.0


def test_nephrology_2():
    assert bench_nephrology_2()["synthetic_nephrology_2"] == 1.0


def test_hepatology_2():
    assert bench_hepatology_2()["synthetic_hepatology_2"] == 1.0
