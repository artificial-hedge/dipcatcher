from quant_fund.models.anticipating_sde import (
    bench_anticipating_sde,
)
from quant_fund.models.delayed_sde import (
    bench_delayed_sde,
)
from quant_fund.models.forward_sde import (
    bench_forward_sde,
)
from quant_fund.models.functional_sde import (
    bench_functional_sde,
)
from quant_fund.models.neutral_sde import (
    bench_neutral_sde,
)
from quant_fund.models.random_sde import (
    bench_random_sde,
)


def test_forward_sde():
    assert bench_forward_sde()["synthetic_forward_sde"] == 1.0


def test_random_sde():
    assert bench_random_sde()["synthetic_random_sde"] == 1.0


def test_anticipating_sde():
    assert bench_anticipating_sde()["synthetic_anticipating_sde"] == 1.0


def test_functional_sde():
    assert bench_functional_sde()["synthetic_functional_sde"] == 1.0


def test_delayed_sde():
    assert bench_delayed_sde()["synthetic_delayed_sde"] == 1.0


def test_neutral_sde():
    assert bench_neutral_sde()["synthetic_neutral_sde"] == 1.0
