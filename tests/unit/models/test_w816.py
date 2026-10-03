from quant_fund.models.cadlag_markov import (
    bench_cadlag_markov,
)
from quant_fund.models.characteristic_markov import (
    bench_characteristic_markov,
)
from quant_fund.models.generator_markov import (
    bench_generator_markov,
)
from quant_fund.models.hunt_process import (
    bench_hunt_process,
)
from quant_fund.models.resolvent_markov import (
    bench_resolvent_markov,
)
from quant_fund.models.transition_semigroup import (
    bench_transition_semigroup,
)


def test_hunt_process():
    assert bench_hunt_process()["synthetic_hunt_process"] == 1.0


def test_cadlag_markov():
    assert bench_cadlag_markov()["synthetic_cadlag_markov"] == 1.0


def test_transition_semigroup():
    assert bench_transition_semigroup()["synthetic_transition_semigroup"] == 1.0


def test_resolvent_markov():
    assert bench_resolvent_markov()["synthetic_resolvent_markov"] == 1.0


def test_generator_markov():
    assert bench_generator_markov()["synthetic_generator_markov"] == 1.0


def test_characteristic_markov():
    assert bench_characteristic_markov()["synthetic_characteristic_markov"] == 1.0
