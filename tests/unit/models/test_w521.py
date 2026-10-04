from quant_fund.models.chebyshev_bias import bench_chebyshev_bias
from quant_fund.models.dirichlet_l import bench_dirichlet_l
from quant_fund.models.explicit_formula import bench_explicit_formula
from quant_fund.models.linnik_thm import bench_linnik_thm
from quant_fund.models.riemann_zeta import bench_riemann_zeta
from quant_fund.models.zero_density import bench_zero_density


def test_explicit_formula():
    assert bench_explicit_formula()["synthetic_explicit_formula"] == 1.0


def test_zero_density():
    assert bench_zero_density()["synthetic_zero_density"] == 1.0


def test_riemann_zeta():
    assert bench_riemann_zeta()["synthetic_riemann_zeta"] == 1.0


def test_dirichlet_l():
    assert bench_dirichlet_l()["synthetic_dirichlet_l"] == 1.0


def test_linnik_thm():
    assert bench_linnik_thm()["synthetic_linnik_thm"] == 1.0


def test_chebyshev_bias():
    assert bench_chebyshev_bias()["synthetic_chebyshev_bias"] == 1.0
