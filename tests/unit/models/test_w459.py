from quant_fund.models.chain_cond import bench_chain_cond
from quant_fund.models.denotational import bench_denotational
from quant_fund.models.fixed_points_ord import bench_fixed_points_ord
from quant_fund.models.galois_insertion import bench_galois_insertion
from quant_fund.models.scott_cpo import bench_scott_cpo
from quant_fund.models.way_below import bench_way_below


def test_fixed_points_ord():
    assert bench_fixed_points_ord()["synthetic_fixed_points_ord"] == 1.0


def test_chain_cond():
    assert bench_chain_cond()["synthetic_chain_cond"] == 1.0


def test_scott_cpo():
    assert bench_scott_cpo()["synthetic_scott_cpo"] == 1.0


def test_way_below():
    assert bench_way_below()["synthetic_way_below"] == 1.0


def test_galois_insertion():
    assert bench_galois_insertion()["synthetic_galois_insertion"] == 1.0


def test_denotational():
    assert bench_denotational()["synthetic_denotational"] == 1.0
