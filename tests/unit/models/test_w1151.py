"""Wave-1151 chemical-sciences canon tests."""

from __future__ import annotations

from quant_fund.models.analytical_chemistry_2 import bench_analytical_chemistry_2
from quant_fund.models.chemistry_3 import bench_chemistry_3
from quant_fund.models.electrochemistry_2 import bench_electrochemistry_2
from quant_fund.models.inorganic_chemistry_2 import bench_inorganic_chemistry_2
from quant_fund.models.organic_chemistry_2 import bench_organic_chemistry_2
from quant_fund.models.physical_chemistry_2 import bench_physical_chemistry_2


def test_chemistry_3():
    assert bench_chemistry_3()["synthetic_chemistry_3"] == 1.0


def test_organic_chemistry_2():
    assert bench_organic_chemistry_2()["synthetic_organic_chemistry_2"] == 1.0


def test_inorganic_chemistry_2():
    assert bench_inorganic_chemistry_2()["synthetic_inorganic_chemistry_2"] == 1.0


def test_physical_chemistry_2():
    assert bench_physical_chemistry_2()["synthetic_physical_chemistry_2"] == 1.0


def test_analytical_chemistry_2():
    assert bench_analytical_chemistry_2()["synthetic_analytical_chemistry_2"] == 1.0


def test_electrochemistry_2():
    assert bench_electrochemistry_2()["synthetic_electrochemistry_2"] == 1.0
