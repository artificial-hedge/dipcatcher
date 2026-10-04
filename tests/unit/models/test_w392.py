from quant_fund.models.arithmetization import bench_arithmetization
from quant_fund.models.diagonal_lemma import bench_diagonal_lemma
from quant_fund.models.fixed_point_combinator import bench_fixed_point_combinator
from quant_fund.models.kleene_normal import bench_kleene_normal
from quant_fund.models.mu_recursion import bench_mu_recursion
from quant_fund.models.primitive_recursion import bench_primitive_recursion


def test_mu_recursion():
    assert bench_mu_recursion()["synthetic_mu_recursion"] == 1.0


def test_primitive_recursion():
    assert bench_primitive_recursion()["synthetic_primitive_recursion"] == 1.0


def test_diagonal_lemma():
    assert bench_diagonal_lemma()["synthetic_diagonal_lemma"] == 1.0


def test_arithmetization():
    assert bench_arithmetization()["synthetic_arithmetization"] == 1.0


def test_fixed_point_combinator():
    assert bench_fixed_point_combinator()["synthetic_fixed_point_combinator"] == 1.0


def test_kleene_normal():
    assert bench_kleene_normal()["synthetic_kleene_normal"] == 1.0
