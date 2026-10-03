"""Wave-1005 general-relativity canon tests."""

from __future__ import annotations

from quant_fund.models.einstein_equations import bench_einstein_equations
from quant_fund.models.friedmann_eq import bench_friedmann_eq
from quant_fund.models.gr_birkhoff import bench_gr_birkhoff
from quant_fund.models.kerr_metric import bench_kerr_metric
from quant_fund.models.penrose_diagrams import bench_penrose_diagrams
from quant_fund.models.schwarzschild_metric import bench_schwarzschild_metric


def test_einstein_equations():
    assert bench_einstein_equations()["synthetic_einstein_equations"] == 1.0


def test_schwarzschild_metric():
    assert bench_schwarzschild_metric()["synthetic_schwarzschild_metric"] == 1.0


def test_friedmann_eq():
    assert bench_friedmann_eq()["synthetic_friedmann_eq"] == 1.0


def test_kerr_metric():
    assert bench_kerr_metric()["synthetic_kerr_metric"] == 1.0


def test_gr_birkhoff():
    assert bench_gr_birkhoff()["synthetic_gr_birkhoff"] == 1.0


def test_penrose_diagrams():
    assert bench_penrose_diagrams()["synthetic_penrose_diagrams"] == 1.0
