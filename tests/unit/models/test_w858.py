from quant_fund.models.adaptive_simpsons import (
    bench_adaptive_simpsons,
)
from quant_fund.models.double_exp_quad import (
    bench_double_exp_quad,
)
from quant_fund.models.filon_quad import (
    bench_filon_quad,
)
from quant_fund.models.levin_quad import (
    bench_levin_quad,
)
from quant_fund.models.osc_singular import (
    bench_osc_singular,
)
from quant_fund.models.tanh_sinh import (
    bench_tanh_sinh,
)


def test_adaptive_simpsons():
    assert bench_adaptive_simpsons()["synthetic_adaptive_simpsons"] == 1.0


def test_tanh_sinh():
    assert bench_tanh_sinh()["synthetic_tanh_sinh"] == 1.0


def test_double_exp_quad():
    assert bench_double_exp_quad()["synthetic_double_exp_quad"] == 1.0


def test_osc_singular():
    assert bench_osc_singular()["synthetic_osc_singular"] == 1.0


def test_filon_quad():
    assert bench_filon_quad()["synthetic_filon_quad"] == 1.0


def test_levin_quad():
    assert bench_levin_quad()["synthetic_levin_quad"] == 1.0
