"""Wave-932 local-search-3 canon tests."""

from __future__ import annotations

from quant_fund.models.guided_local import bench_guided_local
from quant_fund.models.large_neighborhood import bench_large_neighborhood
from quant_fund.models.path_relinking import bench_path_relinking
from quant_fund.models.ruin_recreate import bench_ruin_recreate
from quant_fund.models.simulated_annealing import bench_simulated_annealing
from quant_fund.models.vns_search import bench_vns_search


def test_vns_search():
    assert bench_vns_search()["synthetic_vns_search"] == 1.0


def test_large_neighborhood():
    assert bench_large_neighborhood()["synthetic_large_neighborhood"] == 1.0


def test_ruin_recreate():
    assert bench_ruin_recreate()["synthetic_ruin_recreate"] == 1.0


def test_path_relinking():
    assert bench_path_relinking()["synthetic_path_relinking"] == 1.0


def test_guided_local():
    assert bench_guided_local()["synthetic_guided_local"] == 1.0


def test_simulated_annealing():
    assert bench_simulated_annealing()["synthetic_simulated_annealing"] == 1.0
