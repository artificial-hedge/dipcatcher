from quant_fund.models.ancient_solution import bench_ancient_solution
from quant_fund.models.hamilton_ricci import bench_hamilton_ricci
from quant_fund.models.kahler_ricci_flow import bench_kahler_ricci_flow
from quant_fund.models.mean_curvature_flow import (
    bench_mean_curvature_flow,
)
from quant_fund.models.perelman_entropy import bench_perelman_entropy
from quant_fund.models.ricci_soliton import bench_ricci_soliton


def test_hamilton_ricci():
    assert bench_hamilton_ricci()["synthetic_hamilton_ricci"] == 1.0


def test_perelman_entropy():
    assert bench_perelman_entropy()["synthetic_perelman_entropy"] == 1.0


def test_ricci_soliton():
    assert bench_ricci_soliton()["synthetic_ricci_soliton"] == 1.0


def test_kahler_ricci_flow():
    assert bench_kahler_ricci_flow()["synthetic_kahler_ricci_flow"] == 1.0


def test_mean_curvature_flow():
    assert bench_mean_curvature_flow()["synthetic_mean_curvature_flow"] == 1.0


def test_ancient_solution():
    assert bench_ancient_solution()["synthetic_ancient_solution"] == 1.0
