from quant_fund.models.chebyshev_u import (
    bench_chebyshev_u,
)
from quant_fund.models.epsilon_algo import (
    bench_epsilon_algo,
)
from quant_fund.models.hexahedral_basis import (
    bench_hexahedral_basis,
)
from quant_fund.models.quadrilateral_basis import (
    bench_quadrilateral_basis,
)
from quant_fund.models.spline_theory import (
    bench_spline_theory,
)
from quant_fund.models.walsh_table import (
    bench_walsh_table,
)


def test_quadrilateral_basis():
    assert bench_quadrilateral_basis()["synthetic_quadrilateral_basis"] == 1.0


def test_hexahedral_basis():
    assert bench_hexahedral_basis()["synthetic_hexahedral_basis"] == 1.0


def test_chebyshev_u():
    assert bench_chebyshev_u()["synthetic_chebyshev_u"] == 1.0


def test_walsh_table():
    assert bench_walsh_table()["synthetic_walsh_table"] == 1.0


def test_epsilon_algo():
    assert bench_epsilon_algo()["synthetic_epsilon_algo"] == 1.0


def test_spline_theory():
    assert bench_spline_theory()["synthetic_spline_theory"] == 1.0
