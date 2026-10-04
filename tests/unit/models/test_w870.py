from quant_fund.models.augmented_lagrangian import (
    bench_augmented_lagrangian,
)
from quant_fund.models.conjugate_opt import (
    bench_conjugate_opt,
)
from quant_fund.models.grad_descent_nest import (
    bench_grad_descent_nest,
)
from quant_fund.models.interior_point2 import (
    bench_interior_point2,
)
from quant_fund.models.newton_method import (
    bench_newton_method,
)
from quant_fund.models.quasi_newton_lbfgs import (
    bench_quasi_newton_lbfgs,
)


def test_newton_method():
    assert bench_newton_method()["synthetic_newton_method"] == 1.0


def test_quasi_newton_lbfgs():
    assert bench_quasi_newton_lbfgs()["synthetic_quasi_newton_lbfgs"] == 1.0


def test_augmented_lagrangian():
    assert bench_augmented_lagrangian()["synthetic_augmented_lagrangian"] == 1.0


def test_interior_point2():
    assert bench_interior_point2()["synthetic_interior_point2"] == 1.0


def test_grad_descent_nest():
    assert bench_grad_descent_nest()["synthetic_grad_descent_nest"] == 1.0


def test_conjugate_opt():
    assert bench_conjugate_opt()["synthetic_conjugate_opt"] == 1.0
