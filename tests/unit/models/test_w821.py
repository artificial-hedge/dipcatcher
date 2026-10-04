from quant_fund.models.compensator_rm import (
    bench_compensator_rm,
)
from quant_fund.models.integer_measure import (
    bench_integer_measure,
)
from quant_fund.models.jump_measure import (
    bench_jump_measure,
)
from quant_fund.models.poisson_rm import (
    bench_poisson_rm,
)
from quant_fund.models.random_measure import (
    bench_random_measure,
)
from quant_fund.models.sato_measure import (
    bench_sato_measure,
)


def test_random_measure():
    assert bench_random_measure()["synthetic_random_measure"] == 1.0


def test_integer_measure():
    assert bench_integer_measure()["synthetic_integer_measure"] == 1.0


def test_poisson_rm():
    assert bench_poisson_rm()["synthetic_poisson_rm"] == 1.0


def test_compensator_rm():
    assert bench_compensator_rm()["synthetic_compensator_rm"] == 1.0


def test_jump_measure():
    assert bench_jump_measure()["synthetic_jump_measure"] == 1.0


def test_sato_measure():
    assert bench_sato_measure()["synthetic_sato_measure"] == 1.0
