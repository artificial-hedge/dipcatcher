from quant_fund.models.coupled_fbsde import (
    bench_coupled_fbsde,
)
from quant_fund.models.decoupling_field2 import (
    bench_decoupling_field2,
)
from quant_fund.models.four_step_scheme import (
    bench_four_step_scheme,
)
from quant_fund.models.quasi_bsde import (
    bench_quasi_bsde,
)
from quant_fund.models.random_bsde import (
    bench_random_bsde,
)
from quant_fund.models.time_bsde import (
    bench_time_bsde,
)


def test_four_step_scheme():
    assert bench_four_step_scheme()["synthetic_four_step_scheme"] == 1.0


def test_decoupling_field2():
    assert bench_decoupling_field2()["synthetic_decoupling_field2"] == 1.0


def test_quasi_bsde():
    assert bench_quasi_bsde()["synthetic_quasi_bsde"] == 1.0


def test_coupled_fbsde():
    assert bench_coupled_fbsde()["synthetic_coupled_fbsde"] == 1.0


def test_random_bsde():
    assert bench_random_bsde()["synthetic_random_bsde"] == 1.0


def test_time_bsde():
    assert bench_time_bsde()["synthetic_time_bsde"] == 1.0
