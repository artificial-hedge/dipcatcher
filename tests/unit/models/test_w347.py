from quant_fund.models.compact_space import bench_compact_space
from quant_fund.models.connected_space import bench_connected_space
from quant_fund.models.convergence_space import bench_convergence_space
from quant_fund.models.product_topology import bench_product_topology
from quant_fund.models.quotient_topology import bench_quotient_topology
from quant_fund.models.tietze_urysohn import bench_tietze_urysohn


def test_compact_space():
    assert bench_compact_space()["synthetic_compact_space"] == 1.0


def test_connected_space():
    assert bench_connected_space()["synthetic_connected_space"] == 1.0


def test_quotient_topology():
    assert bench_quotient_topology()["synthetic_quotient_topology"] == 1.0


def test_product_topology():
    assert bench_product_topology()["synthetic_product_topology"] == 1.0


def test_convergence_space():
    assert bench_convergence_space()["synthetic_convergence_space"] == 1.0


def test_tietze_urysohn():
    assert bench_tietze_urysohn()["synthetic_tietze_urysohn"] == 1.0
