from quant_fund.models.backward_sde import (
    bench_backward_sde,
)
from quant_fund.models.bsde_solver import (
    bench_bsde_solver,
)
from quant_fund.models.fbsde_markov import (
    bench_fbsde_markov,
)
from quant_fund.models.pardoux_peng import (
    bench_pardoux_peng,
)
from quant_fund.models.reflected_bsde import (
    bench_reflected_bsde,
)
from quant_fund.models.second_order_bsde import (
    bench_second_order_bsde,
)


def test_bsde_solver():
    assert bench_bsde_solver()["synthetic_bsde_solver"] == 1.0


def test_fbsde_markov():
    assert bench_fbsde_markov()["synthetic_fbsde_markov"] == 1.0


def test_backward_sde():
    assert bench_backward_sde()["synthetic_backward_sde"] == 1.0


def test_pardoux_peng():
    assert bench_pardoux_peng()["synthetic_pardoux_peng"] == 1.0


def test_reflected_bsde():
    assert bench_reflected_bsde()["synthetic_reflected_bsde"] == 1.0


def test_second_order_bsde():
    assert bench_second_order_bsde()["synthetic_second_order_bsde"] == 1.0
