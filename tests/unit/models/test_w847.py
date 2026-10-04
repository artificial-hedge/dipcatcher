from quant_fund.models.boundary_layer import (
    bench_boundary_layer,
)
from quant_fund.models.lindstedt_poincare import (
    bench_lindstedt_poincare,
)
from quant_fund.models.matched_asymptotic import (
    bench_matched_asymptotic,
)
from quant_fund.models.multiple_scales import (
    bench_multiple_scales,
)
from quant_fund.models.regular_perturbation import (
    bench_regular_perturbation,
)
from quant_fund.models.singular_perturbation import (
    bench_singular_perturbation,
)


def test_regular_perturbation():
    assert bench_regular_perturbation()["synthetic_regular_perturbation"] == 1.0


def test_singular_perturbation():
    assert bench_singular_perturbation()["synthetic_singular_perturbation"] == 1.0


def test_matched_asymptotic():
    assert bench_matched_asymptotic()["synthetic_matched_asymptotic"] == 1.0


def test_multiple_scales():
    assert bench_multiple_scales()["synthetic_multiple_scales"] == 1.0


def test_lindstedt_poincare():
    assert bench_lindstedt_poincare()["synthetic_lindstedt_poincare"] == 1.0


def test_boundary_layer():
    assert bench_boundary_layer()["synthetic_boundary_layer"] == 1.0
