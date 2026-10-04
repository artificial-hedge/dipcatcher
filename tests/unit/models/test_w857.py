from quant_fund.models.clenshaw_curtis import (
    bench_clenshaw_curtis,
)
from quant_fund.models.fejer_quad import (
    bench_fejer_quad,
)
from quant_fund.models.gauss_chebyshev import (
    bench_gauss_chebyshev,
)
from quant_fund.models.gauss_kronrod import (
    bench_gauss_kronrod,
)
from quant_fund.models.gauss_legendre import (
    bench_gauss_legendre,
)
from quant_fund.models.newton_cotes import (
    bench_newton_cotes,
)


def test_gauss_legendre():
    assert bench_gauss_legendre()["synthetic_gauss_legendre"] == 1.0


def test_gauss_chebyshev():
    assert bench_gauss_chebyshev()["synthetic_gauss_chebyshev"] == 1.0


def test_clenshaw_curtis():
    assert bench_clenshaw_curtis()["synthetic_clenshaw_curtis"] == 1.0


def test_newton_cotes():
    assert bench_newton_cotes()["synthetic_newton_cotes"] == 1.0


def test_gauss_kronrod():
    assert bench_gauss_kronrod()["synthetic_gauss_kronrod"] == 1.0


def test_fejer_quad():
    assert bench_fejer_quad()["synthetic_fejer_quad"] == 1.0
