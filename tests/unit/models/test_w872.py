from quant_fund.models.amg_precond import (
    bench_amg_precond,
)
from quant_fund.models.ic_precond import (
    bench_ic_precond,
)
from quant_fund.models.ilut_precond import (
    bench_ilut_precond,
)
from quant_fund.models.jacobi_precond import (
    bench_jacobi_precond,
)
from quant_fund.models.polynomial_precond import (
    bench_polynomial_precond,
)
from quant_fund.models.ssor_precond import (
    bench_ssor_precond,
)


def test_jacobi_precond():
    assert bench_jacobi_precond()["synthetic_jacobi_precond"] == 1.0


def test_ilut_precond():
    assert bench_ilut_precond()["synthetic_ilut_precond"] == 1.0


def test_ssor_precond():
    assert bench_ssor_precond()["synthetic_ssor_precond"] == 1.0


def test_amg_precond():
    assert bench_amg_precond()["synthetic_amg_precond"] == 1.0


def test_ic_precond():
    assert bench_ic_precond()["synthetic_ic_precond"] == 1.0


def test_polynomial_precond():
    assert bench_polynomial_precond()["synthetic_polynomial_precond"] == 1.0
