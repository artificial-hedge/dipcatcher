from quant_fund.models.dual_matroid import bench_dual_matroid
from quant_fund.models.greedy_matroid import bench_greedy_matroid
from quant_fund.models.matroid_axioms import bench_matroid_axioms
from quant_fund.models.matroid_intersect import bench_matroid_intersect
from quant_fund.models.matroid_union import bench_matroid_union
from quant_fund.models.represented_matroid import bench_represented_matroid


def test_matroid_axioms():
    assert bench_matroid_axioms()["synthetic_matroid_axioms"] == 1.0


def test_greedy_matroid():
    assert bench_greedy_matroid()["synthetic_greedy_matroid"] == 1.0


def test_matroid_intersect():
    assert bench_matroid_intersect()["synthetic_matroid_intersect"] == 1.0


def test_dual_matroid():
    assert bench_dual_matroid()["synthetic_dual_matroid"] == 1.0


def test_matroid_union():
    assert bench_matroid_union()["synthetic_matroid_union"] == 1.0


def test_represented_matroid():
    assert bench_represented_matroid()["synthetic_represented_matroid"] == 1.0
