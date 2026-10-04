"""Wave-939 set-feasibility canon tests."""

from __future__ import annotations

from quant_fund.models.cq_algorithm import bench_cq_algorithm
from quant_fund.models.dykstra_proj import bench_dykstra_proj
from quant_fund.models.halpern_iter import bench_halpern_iter
from quant_fund.models.haugazeau_proj import bench_haugazeau_proj
from quant_fund.models.parallel_prox import bench_parallel_prox
from quant_fund.models.split_feasibility import bench_split_feasibility


def test_split_feasibility():
    assert bench_split_feasibility()["synthetic_split_feasibility"] == 1.0


def test_cq_algorithm():
    assert bench_cq_algorithm()["synthetic_cq_algorithm"] == 1.0


def test_dykstra_proj():
    assert bench_dykstra_proj()["synthetic_dykstra_proj"] == 1.0


def test_haugazeau_proj():
    assert bench_haugazeau_proj()["synthetic_haugazeau_proj"] == 1.0


def test_parallel_prox():
    assert bench_parallel_prox()["synthetic_parallel_prox"] == 1.0


def test_halpern_iter():
    assert bench_halpern_iter()["synthetic_halpern_iter"] == 1.0
