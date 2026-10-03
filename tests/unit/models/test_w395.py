from quant_fund.models.bell_triangle import bench_bell_triangle
from quant_fund.models.catalan_dp import bench_catalan_dp
from quant_fund.models.eulerian_num import bench_eulerian_num
from quant_fund.models.inclusion_excl import bench_inclusion_excl
from quant_fund.models.partition_count import bench_partition_count
from quant_fund.models.stirling_cycle import bench_stirling_cycle


def test_catalan_dp():
    assert bench_catalan_dp()["synthetic_catalan_dp"] == 1.0


def test_stirling_cycle():
    assert bench_stirling_cycle()["synthetic_stirling_cycle"] == 1.0


def test_partition_count():
    assert bench_partition_count()["synthetic_partition_count"] == 1.0


def test_bell_triangle():
    assert bench_bell_triangle()["synthetic_bell_triangle"] == 1.0


def test_eulerian_num():
    assert bench_eulerian_num()["synthetic_eulerian_num"] == 1.0


def test_inclusion_excl():
    assert bench_inclusion_excl()["synthetic_inclusion_excl"] == 1.0
