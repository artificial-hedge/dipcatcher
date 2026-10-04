"""Wave-1100 chemistry canon tests."""

from __future__ import annotations

from quant_fund.models.analytical_chemistry import bench_analytical_chemistry
from quant_fund.models.biochemistry import bench_biochemistry
from quant_fund.models.electrochemistry import bench_electrochemistry
from quant_fund.models.inorganic_chemistry import bench_inorganic_chemistry
from quant_fund.models.organic_chemistry import bench_organic_chemistry
from quant_fund.models.physical_chemistry import bench_physical_chemistry


def test_organic_chemistry():
    assert bench_organic_chemistry()["synthetic_organic_chemistry"] == 1.0


def test_inorganic_chemistry():
    assert bench_inorganic_chemistry()["synthetic_inorganic_chemistry"] == 1.0


def test_physical_chemistry():
    assert bench_physical_chemistry()["synthetic_physical_chemistry"] == 1.0


def test_analytical_chemistry():
    assert bench_analytical_chemistry()["synthetic_analytical_chemistry"] == 1.0


def test_biochemistry():
    assert bench_biochemistry()["synthetic_biochemistry"] == 1.0


def test_electrochemistry():
    assert bench_electrochemistry()["synthetic_electrochemistry"] == 1.0
