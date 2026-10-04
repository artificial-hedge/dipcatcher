from quant_fund.models.conditional_expect import bench_conditional_expect
from quant_fund.models.conv_sum import bench_conv_sum
from quant_fund.models.kolmogorov_axioms import bench_kolmogorov_axioms
from quant_fund.models.markov_ineq import bench_markov_ineq
from quant_fund.models.moment_generating import bench_moment_generating
from quant_fund.models.stochastic_order import bench_stochastic_order


def test_kolmogorov_axioms():
    assert bench_kolmogorov_axioms()["synthetic_kolmogorov_axioms"] == 1.0


def test_conditional_expect():
    assert bench_conditional_expect()["synthetic_conditional_expect"] == 1.0


def test_markov_ineq():
    assert bench_markov_ineq()["synthetic_markov_ineq"] == 1.0


def test_conv_sum():
    assert bench_conv_sum()["synthetic_conv_sum"] == 1.0


def test_moment_generating():
    assert bench_moment_generating()["synthetic_moment_generating"] == 1.0


def test_stochastic_order():
    assert bench_stochastic_order()["synthetic_stochastic_order"] == 1.0
