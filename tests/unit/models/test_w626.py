from quant_fund.models.algebraic_stack2 import (
    bench_algebraic_stack2,
)
from quant_fund.models.artin_stack import bench_artin_stack
from quant_fund.models.gerbe_cohomology import (
    bench_gerbe_cohomology,
)
from quant_fund.models.orbifold_stack import (
    bench_orbifold_stack,
)
from quant_fund.models.quotient_stack2 import (
    bench_quotient_stack2,
)
from quant_fund.models.stacky_point import (
    bench_stacky_point,
)


def test_algebraic_stack2():
    assert bench_algebraic_stack2()["synthetic_algebraic_stack2"] == 1.0


def test_artin_stack():
    assert bench_artin_stack()["synthetic_artin_stack"] == 1.0


def test_quotient_stack2():
    assert bench_quotient_stack2()["synthetic_quotient_stack2"] == 1.0


def test_stacky_point():
    assert bench_stacky_point()["synthetic_stacky_point"] == 1.0


def test_orbifold_stack():
    assert bench_orbifold_stack()["synthetic_orbifold_stack"] == 1.0


def test_gerbe_cohomology():
    assert bench_gerbe_cohomology()["synthetic_gerbe_cohomology"] == 1.0
