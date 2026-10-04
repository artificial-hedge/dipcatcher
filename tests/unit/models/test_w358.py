from quant_fund.models.banach_alaoglu import bench_banach_alaoglu
from quant_fund.models.closed_graph import bench_closed_graph
from quant_fund.models.open_mapping import bench_open_mapping
from quant_fund.models.reflexive_space import bench_reflexive_space
from quant_fund.models.uniform_bounded import bench_uniform_bounded
from quant_fund.models.weak_convergence import bench_weak_convergence


def test_open_mapping():
    assert bench_open_mapping()["synthetic_open_mapping"] == 1.0


def test_uniform_bounded():
    assert bench_uniform_bounded()["synthetic_uniform_bounded"] == 1.0


def test_weak_convergence():
    assert bench_weak_convergence()["synthetic_weak_convergence"] == 1.0


def test_banach_alaoglu():
    assert bench_banach_alaoglu()["synthetic_banach_alaoglu"] == 1.0


def test_reflexive_space():
    assert bench_reflexive_space()["synthetic_reflexive_space"] == 1.0


def test_closed_graph():
    assert bench_closed_graph()["synthetic_closed_graph"] == 1.0
