from quant_fund.models.burnside_lemma import bench_burnside_lemma
from quant_fund.models.cayley_graph import bench_cayley_graph
from quant_fund.models.conjugacy_classes import bench_conjugacy_classes
from quant_fund.models.free_group import bench_free_group
from quant_fund.models.group_presentation import bench_group_presentation
from quant_fund.models.sylow_theorems import bench_sylow_theorems


def test_sylow_theorems():
    assert bench_sylow_theorems()["synthetic_sylow_theorems"] == 1.0


def test_group_presentation():
    assert bench_group_presentation()["synthetic_group_presentation"] == 1.0


def test_burnside_lemma():
    assert bench_burnside_lemma()["synthetic_burnside_lemma"] == 1.0


def test_free_group():
    assert bench_free_group()["synthetic_free_group"] == 1.0


def test_conjugacy_classes():
    assert bench_conjugacy_classes()["synthetic_conjugacy_classes"] == 1.0


def test_cayley_graph():
    assert bench_cayley_graph()["synthetic_cayley_graph"] == 1.0
