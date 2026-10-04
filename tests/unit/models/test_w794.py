from quant_fund.models.dynamic_programming import (
    bench_dynamic_programming,
)
from quant_fund.models.hamilton_jacobi import (
    bench_hamilton_jacobi,
)
from quant_fund.models.impulsive_control import (
    bench_impulsive_control,
)
from quant_fund.models.quasi_variational import (
    bench_quasi_variational,
)
from quant_fund.models.verification_thm import (
    bench_verification_thm,
)
from quant_fund.models.viscosity_solution import (
    bench_viscosity_solution,
)


def test_dynamic_programming():
    assert bench_dynamic_programming()["synthetic_dynamic_programming"] == 1.0


def test_verification_thm():
    assert bench_verification_thm()["synthetic_verification_thm"] == 1.0


def test_hamilton_jacobi():
    assert bench_hamilton_jacobi()["synthetic_hamilton_jacobi"] == 1.0


def test_viscosity_solution():
    assert bench_viscosity_solution()["synthetic_viscosity_solution"] == 1.0


def test_quasi_variational():
    assert bench_quasi_variational()["synthetic_quasi_variational"] == 1.0


def test_impulsive_control():
    assert bench_impulsive_control()["synthetic_impulsive_control"] == 1.0
