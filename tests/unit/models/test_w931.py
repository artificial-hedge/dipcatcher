"""Wave-931 population-metaheuristics canon tests."""

from __future__ import annotations

from quant_fund.models.ant_colony import bench_ant_colony
from quant_fund.models.diff_evolution import bench_diff_evolution
from quant_fund.models.firefly_algo import bench_firefly_algo
from quant_fund.models.genetic_tsp import bench_genetic_tsp
from quant_fund.models.harmony_search import bench_harmony_search
from quant_fund.models.pso_swarm import bench_pso_swarm


def test_ant_colony():
    assert bench_ant_colony()["synthetic_ant_colony"] == 1.0


def test_pso_swarm():
    assert bench_pso_swarm()["synthetic_pso_swarm"] == 1.0


def test_diff_evolution():
    assert bench_diff_evolution()["synthetic_diff_evolution"] == 1.0


def test_genetic_tsp():
    assert bench_genetic_tsp()["synthetic_genetic_tsp"] == 1.0


def test_firefly_algo():
    assert bench_firefly_algo()["synthetic_firefly_algo"] == 1.0


def test_harmony_search():
    assert bench_harmony_search()["synthetic_harmony_search"] == 1.0
