from quant_fund.models.degree_mod2 import bench_degree_mod2
from quant_fund.models.handle_decomp import bench_handle_decomp
from quant_fund.models.morse_theory import bench_morse_theory
from quant_fund.models.poincare_hopf import bench_poincare_hopf
from quant_fund.models.regular_value import bench_regular_value
from quant_fund.models.transversality import bench_transversality


def test_morse_theory():
    assert bench_morse_theory()["synthetic_morse_theory"] == 1.0


def test_transversality():
    assert bench_transversality()["synthetic_transversality"] == 1.0


def test_regular_value():
    assert bench_regular_value()["synthetic_regular_value"] == 1.0


def test_degree_mod2():
    assert bench_degree_mod2()["synthetic_degree_mod2"] == 1.0


def test_handle_decomp():
    assert bench_handle_decomp()["synthetic_handle_decomp"] == 1.0


def test_poincare_hopf():
    assert bench_poincare_hopf()["synthetic_poincare_hopf"] == 1.0
