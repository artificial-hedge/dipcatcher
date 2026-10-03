from quant_fund.models.differential_game import (
    bench_differential_game,
)
from quant_fund.models.dynkin_game import (
    bench_dynkin_game,
)
from quant_fund.models.isaacs_equation import (
    bench_isaacs_equation,
)
from quant_fund.models.nonzero_sum_game import (
    bench_nonzero_sum_game,
)
from quant_fund.models.stochastic_game2 import (
    bench_stochastic_game2,
)
from quant_fund.models.zero_sum_game import (
    bench_zero_sum_game,
)


def test_dynkin_game():
    assert bench_dynkin_game()["synthetic_dynkin_game"] == 1.0


def test_stochastic_game2():
    assert bench_stochastic_game2()["synthetic_stochastic_game2"] == 1.0


def test_differential_game():
    assert bench_differential_game()["synthetic_differential_game"] == 1.0


def test_zero_sum_game():
    assert bench_zero_sum_game()["synthetic_zero_sum_game"] == 1.0


def test_nonzero_sum_game():
    assert bench_nonzero_sum_game()["synthetic_nonzero_sum_game"] == 1.0


def test_isaacs_equation():
    assert bench_isaacs_equation()["synthetic_isaacs_equation"] == 1.0
