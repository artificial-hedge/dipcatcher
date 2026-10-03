from quant_fund.models.doubly_bsde import (
    bench_doubly_bsde,
)
from quant_fund.models.obstacle_bsde import (
    bench_obstacle_bsde,
)
from quant_fund.models.quadratic_bsde import (
    bench_quadratic_bsde,
)
from quant_fund.models.reflected_bsde2 import (
    bench_reflected_bsde2,
)
from quant_fund.models.second_bsde import (
    bench_second_bsde,
)
from quant_fund.models.super_linear import (
    bench_super_linear,
)


def test_second_bsde():
    assert bench_second_bsde()["synthetic_second_bsde"] == 1.0


def test_doubly_bsde():
    assert bench_doubly_bsde()["synthetic_doubly_bsde"] == 1.0


def test_reflected_bsde2():
    assert bench_reflected_bsde2()["synthetic_reflected_bsde2"] == 1.0


def test_obstacle_bsde():
    assert bench_obstacle_bsde()["synthetic_obstacle_bsde"] == 1.0


def test_quadratic_bsde():
    assert bench_quadratic_bsde()["synthetic_quadratic_bsde"] == 1.0


def test_super_linear():
    assert bench_super_linear()["synthetic_super_linear"] == 1.0
