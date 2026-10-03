from quant_fund.models.d_bar_neumann import bench_d_bar_neumann
from quant_fund.models.domain_holo import bench_domain_holo
from quant_fund.models.hartogs_thm import bench_hartogs_thm
from quant_fund.models.levi_problem import bench_levi_problem
from quant_fund.models.oka_coherence import bench_oka_coherence
from quant_fund.models.pseudoconvex import bench_pseudoconvex


def test_hartogs_thm():
    assert bench_hartogs_thm()["synthetic_hartogs_thm"] == 1.0


def test_domain_holo():
    assert bench_domain_holo()["synthetic_domain_holo"] == 1.0


def test_pseudoconvex():
    assert bench_pseudoconvex()["synthetic_pseudoconvex"] == 1.0


def test_levi_problem():
    assert bench_levi_problem()["synthetic_levi_problem"] == 1.0


def test_oka_coherence():
    assert bench_oka_coherence()["synthetic_oka_coherence"] == 1.0


def test_d_bar_neumann():
    assert bench_d_bar_neumann()["synthetic_d_bar_neumann"] == 1.0
