from quant_fund.models.bisimulation import bench_bisimulation
from quant_fund.models.covering_space import bench_covering_space
from quant_fund.models.ef_game import bench_ef_game
from quant_fund.models.fundamental_group import bench_fundamental_group
from quant_fund.models.kripke_semantics import bench_kripke_semantics
from quant_fund.models.topo_separation import bench_topo_separation


def test_kripke_semantics():
    assert bench_kripke_semantics()["synthetic_kripke_semantics"] == 1.0


def test_bisimulation():
    assert bench_bisimulation()["synthetic_bisimulation"] == 1.0


def test_ef_game():
    assert bench_ef_game()["synthetic_ef_game"] == 1.0


def test_fundamental_group():
    assert bench_fundamental_group()["synthetic_fundamental_group"] == 1.0


def test_covering_space():
    assert bench_covering_space()["synthetic_covering_space"] == 1.0


def test_topo_separation():
    assert bench_topo_separation()["synthetic_topo_separation"] == 1.0
