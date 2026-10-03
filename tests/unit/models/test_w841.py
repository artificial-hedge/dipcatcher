from quant_fund.models.chebyshev_t import (
    bench_chebyshev_t,
)
from quant_fund.models.gegenbauer_poly import (
    bench_gegenbauer_poly,
)
from quant_fund.models.hermite_poly import (
    bench_hermite_poly,
)
from quant_fund.models.jacobi_poly import (
    bench_jacobi_poly,
)
from quant_fund.models.laguerre_poly import (
    bench_laguerre_poly,
)
from quant_fund.models.legendre_poly import (
    bench_legendre_poly,
)


def test_legendre_poly():
    assert bench_legendre_poly()["synthetic_legendre_poly"] == 1.0


def test_chebyshev_t():
    assert bench_chebyshev_t()["synthetic_chebyshev_t"] == 1.0


def test_hermite_poly():
    assert bench_hermite_poly()["synthetic_hermite_poly"] == 1.0


def test_laguerre_poly():
    assert bench_laguerre_poly()["synthetic_laguerre_poly"] == 1.0


def test_jacobi_poly():
    assert bench_jacobi_poly()["synthetic_jacobi_poly"] == 1.0


def test_gegenbauer_poly():
    assert bench_gegenbauer_poly()["synthetic_gegenbauer_poly"] == 1.0
