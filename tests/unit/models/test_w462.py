from quant_fund.models.cartesian_fib2 import bench_cartesian_fib2
from quant_fund.models.cohesive_struct import bench_cohesive_struct
from quant_fund.models.descent_cond import bench_descent_cond
from quant_fund.models.lex_reflect import bench_lex_reflect
from quant_fund.models.n_localic import bench_n_localic
from quant_fund.models.shape_theory import bench_shape_theory


def test_n_localic():
    assert bench_n_localic()["synthetic_n_localic"] == 1.0


def test_shape_theory():
    assert bench_shape_theory()["synthetic_shape_theory"] == 1.0


def test_descent_cond():
    assert bench_descent_cond()["synthetic_descent_cond"] == 1.0


def test_lex_reflect():
    assert bench_lex_reflect()["synthetic_lex_reflect"] == 1.0


def test_cartesian_fib2():
    assert bench_cartesian_fib2()["synthetic_cartesian_fib2"] == 1.0


def test_cohesive_struct():
    assert bench_cohesive_struct()["synthetic_cohesive_struct"] == 1.0
