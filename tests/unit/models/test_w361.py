from quant_fund.models.chain_homotopy import bench_chain_homotopy
from quant_fund.models.covering_lift import bench_covering_lift
from quant_fund.models.degree_map import bench_degree_map
from quant_fund.models.euler_homology import bench_euler_homology
from quant_fund.models.homotopy_pi1 import bench_homotopy_pi1
from quant_fund.models.simplicial_homology import bench_simplicial_homology


def test_homotopy_pi1():
    assert bench_homotopy_pi1()["synthetic_homotopy_pi1"] == 1.0


def test_simplicial_homology():
    assert bench_simplicial_homology()["synthetic_simplicial_homology"] == 1.0


def test_chain_homotopy():
    assert bench_chain_homotopy()["synthetic_chain_homotopy"] == 1.0


def test_euler_homology():
    assert bench_euler_homology()["synthetic_euler_homology"] == 1.0


def test_degree_map():
    assert bench_degree_map()["synthetic_degree_map"] == 1.0


def test_covering_lift():
    assert bench_covering_lift()["synthetic_covering_lift"] == 1.0
