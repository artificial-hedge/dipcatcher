from quant_fund.models.breuil_prism import bench_breuil_prism
from quant_fund.models.cartier_prism import bench_cartier_prism
from quant_fund.models.filtered_prism import bench_filtered_prism
from quant_fund.models.frobenius_prism import bench_frobenius_prism
from quant_fund.models.prism_site2 import bench_prism_site2
from quant_fund.models.stacky_prism import bench_stacky_prism


def test_prism_site2():
    assert bench_prism_site2()["synthetic_prism_site2"] == 1.0


def test_cartier_prism():
    assert bench_cartier_prism()["synthetic_cartier_prism"] == 1.0


def test_breuil_prism():
    assert bench_breuil_prism()["synthetic_breuil_prism"] == 1.0


def test_filtered_prism():
    assert bench_filtered_prism()["synthetic_filtered_prism"] == 1.0


def test_frobenius_prism():
    assert bench_frobenius_prism()["synthetic_frobenius_prism"] == 1.0


def test_stacky_prism():
    assert bench_stacky_prism()["synthetic_stacky_prism"] == 1.0
