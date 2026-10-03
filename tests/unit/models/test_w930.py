"""Wave-930 metaheuristics canon tests."""

from __future__ import annotations

from quant_fund.models.grasp_meta import bench_grasp_meta
from quant_fund.models.iterated_local import bench_iterated_local
from quant_fund.models.lin_kernighan import bench_lin_kernighan
from quant_fund.models.tabu_search import bench_tabu_search
from quant_fund.models.three_opt_move import bench_three_opt_move
from quant_fund.models.two_opt_move import bench_two_opt_move


def test_lin_kernighan():
    assert bench_lin_kernighan()["synthetic_lin_kernighan"] == 1.0


def test_two_opt_move():
    assert bench_two_opt_move()["synthetic_two_opt_move"] == 1.0


def test_three_opt_move():
    assert bench_three_opt_move()["synthetic_three_opt_move"] == 1.0


def test_tabu_search():
    assert bench_tabu_search()["synthetic_tabu_search"] == 1.0


def test_iterated_local():
    assert bench_iterated_local()["synthetic_iterated_local"] == 1.0


def test_grasp_meta():
    assert bench_grasp_meta()["synthetic_grasp_meta"] == 1.0
