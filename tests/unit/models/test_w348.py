from quant_fund.models.euler_trail import bench_euler_trail
from quant_fund.models.graph_coloring import bench_graph_coloring
from quant_fund.models.matroid_greedy import bench_matroid_greedy
from quant_fund.models.planar_check import bench_planar_check
from quant_fund.models.poset_dimension import bench_poset_dimension
from quant_fund.models.ramsey_r33 import bench_ramsey_r33


def test_graph_coloring():
    assert bench_graph_coloring()["synthetic_graph_coloring"] == 1.0


def test_euler_trail():
    assert bench_euler_trail()["synthetic_euler_trail"] == 1.0


def test_matroid_greedy():
    assert bench_matroid_greedy()["synthetic_matroid_greedy"] == 1.0


def test_planar_check():
    assert bench_planar_check()["synthetic_planar_check"] == 1.0


def test_poset_dimension():
    assert bench_poset_dimension()["synthetic_poset_dimension"] == 1.0


def test_ramsey_r33():
    assert bench_ramsey_r33()["synthetic_ramsey_r33"] == 1.0
