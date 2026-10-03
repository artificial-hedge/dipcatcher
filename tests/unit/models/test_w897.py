from quant_fund.models.dahlquist_test import (
    bench_dahlquist_test,
)
from quant_fund.models.explicit_midpoint import (
    bench_explicit_midpoint,
)
from quant_fund.models.heun_method import (
    bench_heun_method,
)
from quant_fund.models.linear_multistep import (
    bench_linear_multistep,
)
from quant_fund.models.order_barrier import (
    bench_order_barrier,
)
from quant_fund.models.trapezoid_rule import (
    bench_trapezoid_rule,
)


def test_linear_multistep():
    assert bench_linear_multistep()["synthetic_linear_multistep"] == 1.0


def test_dahlquist_test():
    assert bench_dahlquist_test()["synthetic_dahlquist_test"] == 1.0


def test_explicit_midpoint():
    assert bench_explicit_midpoint()["synthetic_explicit_midpoint"] == 1.0


def test_heun_method():
    assert bench_heun_method()["synthetic_heun_method"] == 1.0


def test_trapezoid_rule():
    assert bench_trapezoid_rule()["synthetic_trapezoid_rule"] == 1.0


def test_order_barrier():
    assert bench_order_barrier()["synthetic_order_barrier"] == 1.0
