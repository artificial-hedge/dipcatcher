from quant_fund.models.cartesian_morphism import (
    bench_cartesian_morphism,
)
from quant_fund.models.fib_infty import bench_fib_infty
from quant_fund.models.infty_functor import (
    bench_infty_functor,
)
from quant_fund.models.inner_horn import bench_inner_horn
from quant_fund.models.joyal_horn import bench_joyal_horn
from quant_fund.models.quasi_cat2 import bench_quasi_cat2


def test_quasi_cat2():
    assert bench_quasi_cat2()["synthetic_quasi_cat2"] == 1.0


def test_inner_horn():
    assert bench_inner_horn()["synthetic_inner_horn"] == 1.0


def test_joyal_horn():
    assert bench_joyal_horn()["synthetic_joyal_horn"] == 1.0


def test_fib_infty():
    assert bench_fib_infty()["synthetic_fib_infty"] == 1.0


def test_cartesian_morphism():
    assert bench_cartesian_morphism()["synthetic_cartesian_morphism"] == 1.0


def test_infty_functor():
    assert bench_infty_functor()["synthetic_infty_functor"] == 1.0
